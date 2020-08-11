//===-- IPO/OpenMPOpt.cpp - Collection of OpenMP specific optimizations ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpenMP specific optimizations:
//
// - Deduplication of runtime calls, e.g., omp_get_thread_num.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/OpenMPOpt.h"

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/AliasAnalysis.h"

using namespace llvm;
using namespace omp;
using namespace types;

#define DEBUG_TYPE "openmp-opt"

static cl::opt<bool> DisableOpenMPOptimizations(
    "openmp-opt-disable", cl::ZeroOrMore,
    cl::desc("Disable OpenMP specific optimizations."), cl::Hidden,
    cl::init(false));

static cl::opt<bool> PrintICVValues("openmp-print-icv-values", cl::init(false),
                                    cl::Hidden);

static cl::opt<bool> SplitMemoryTransfers(
    "openmp-split-memtransfers",
    cl::desc("Tries to hide the latency of host to device memory transfers"),
    cl::Hidden, cl::init(false));

STATISTIC(NumOpenMPRuntimeCallsDeduplicated,
          "Number of OpenMP runtime calls deduplicated");
STATISTIC(NumOpenMPParallelRegionsDeleted,
          "Number of OpenMP parallel regions deleted");
STATISTIC(NumOpenMPRuntimeFunctionsIdentified,
          "Number of OpenMP runtime functions identified");
STATISTIC(NumOpenMPRuntimeFunctionUsesIdentified,
          "Number of OpenMP runtime function uses identified");

#if !defined(NDEBUG)
static constexpr auto TAG = "[" DEBUG_TYPE "]";
#endif

/// Helper struct to store tracked ICV values at specif instructions.
struct ICVValue {
  Instruction *Inst;
  Value *TrackedValue;

  ICVValue(Instruction *I, Value *Val) : Inst(I), TrackedValue(Val) {}
};

namespace llvm {

// Provide DenseMapInfo for ICVValue
template <> struct DenseMapInfo<ICVValue> {
  using InstInfo = DenseMapInfo<Instruction *>;
  using ValueInfo = DenseMapInfo<Value *>;

  static inline ICVValue getEmptyKey() {
    return ICVValue(InstInfo::getEmptyKey(), ValueInfo::getEmptyKey());
  };

  static inline ICVValue getTombstoneKey() {
    return ICVValue(InstInfo::getTombstoneKey(), ValueInfo::getTombstoneKey());
  };

  static unsigned getHashValue(const ICVValue &ICVVal) {
    return detail::combineHashValue(
        InstInfo::getHashValue(ICVVal.Inst),
        ValueInfo::getHashValue(ICVVal.TrackedValue));
  }

  static bool isEqual(const ICVValue &LHS, const ICVValue &RHS) {
    return InstInfo::isEqual(LHS.Inst, RHS.Inst) &&
           ValueInfo::isEqual(LHS.TrackedValue, RHS.TrackedValue);
  }
};

} // end namespace llvm

//===----------------------------------------------------------------------===//
// Definitions of the OMPInformationCache helper structure.
//===----------------------------------------------------------------------===//

using MemoryTransfer = OMPInformationCache::MemoryTransfer;
using OffloadArray = MemoryTransfer::OffloadArray;

void OMPInformationCache::RuntimeFunctionInfo::foreachUse(
    function_ref<bool(Use &, Function &)> CB, Function *F, UseVector *Uses) {
  SmallVector<unsigned, 8> ToBeDeleted;
  ToBeDeleted.clear();

  unsigned Idx = 0;
  UseVector &UV = Uses ? *Uses : getOrCreateUseVector(F);

  for (Use *U : UV) {
    if (CB(*U, *F))
      ToBeDeleted.push_back(Idx);
    ++Idx;
  }

  // Remove the to-be-deleted indices in reverse order as prior
  // modifcations will not modify the smaller indices.
  while (!ToBeDeleted.empty()) {
    unsigned Idx = ToBeDeleted.pop_back_val();
    UV[Idx] = UV.back();
    UV.pop_back();
  }
}

void OMPInformationCache::initializeInternalControlVars() {
#define ICV_RT_SET(_Name, RTL)                                                 \
  {                                                                            \
    auto &ICV = ICVs[_Name];                                                   \
    ICV.Setter = RTL;                                                          \
  }
#define ICV_RT_GET(Name, RTL)                                                  \
  {                                                                            \
    auto &ICV = ICVs[Name];                                                    \
    ICV.Getter = RTL;                                                          \
  }
#define ICV_DATA_ENV(Enum, _Name, _EnvVarName, Init)                           \
  {                                                                            \
    auto &ICV = ICVs[Enum];                                                    \
    ICV.Name = _Name;                                                          \
    ICV.Kind = Enum;                                                           \
    ICV.InitKind = Init;                                                       \
    ICV.EnvVarName = _EnvVarName;                                              \
    switch (ICV.InitKind) {                                                    \
    case ICV_IMPLEMENTATION_DEFINED:                                           \
      ICV.InitValue = nullptr;                                                 \
      break;                                                                   \
    case ICV_ZERO:                                                             \
      ICV.InitValue =                                                          \
          ConstantInt::get(Type::getInt32Ty(Int32->getContext()), 0);          \
      break;                                                                   \
    case ICV_FALSE:                                                            \
      ICV.InitValue = ConstantInt::getFalse(Int1->getContext());               \
      break;                                                                   \
    case ICV_LAST:                                                             \
      break;                                                                   \
    }                                                                          \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"
}

void OMPInformationCache::initializeRuntimeFunctions() {
  // Helper to collect all uses of the decleration in the UsesMap.
  auto CollectUses = [&](RuntimeFunctionInfo &RFI) {
    unsigned NumUses = 0;
    if (!RFI.Declaration)
      return NumUses;
    OMPBuilder.addAttributes(RFI.Kind, *RFI.Declaration);

    NumOpenMPRuntimeFunctionsIdentified += 1;
    NumOpenMPRuntimeFunctionUsesIdentified += RFI.Declaration->getNumUses();

    // TODO: We directly convert uses into proper calls and unknown uses.
    for (Use &U : RFI.Declaration->uses()) {
      if (Instruction *UserI = dyn_cast<Instruction>(U.getUser())) {
        if (ModuleSlice.count(UserI->getFunction())) {
          RFI.getOrCreateUseVector(UserI->getFunction()).push_back(&U);
          ++NumUses;
        }
      } else {
        RFI.getOrCreateUseVector(nullptr).push_back(&U);
        ++NumUses;
      }
    }
    return NumUses;
  };

  Module &M = *((*ModuleSlice.begin())->getParent());

#define OMP_RTL(_Enum, _Name, _IsVarArg, _ReturnType, ...)                     \
  {                                                                            \
    SmallVector<Type *, 8> ArgsTypes({__VA_ARGS__});                           \
    Function *F = M.getFunction(_Name);                                        \
    if (declMatchesRTFTypes(F, _ReturnType, ArgsTypes)) {                      \
      auto &RFI = RFIs[_Enum];                                                 \
      RFI.Kind = _Enum;                                                        \
      RFI.Name = _Name;                                                        \
      RFI.IsVarArg = _IsVarArg;                                                \
      RFI.ReturnType = _ReturnType;                                            \
      RFI.ArgumentTypes = std::move(ArgsTypes);                                \
      RFI.Declaration = F;                                                     \
      unsigned NumUses = CollectUses(RFI);                                     \
      (void)NumUses;                                                           \
      LLVM_DEBUG({                                                             \
        dbgs() << TAG << RFI.Name << (RFI.Declaration ? "" : " not")           \
               << " found\n";                                                  \
        if (RFI.Declaration)                                                   \
          dbgs() << TAG << "-> got " << NumUses << " uses in "                 \
                 << RFI.getNumFunctionsWithUses()                              \
                 << " different functions.\n";                                 \
      });                                                                      \
    }                                                                          \
  }
#include "llvm/Frontend/OpenMP/OMPKinds.def"

  // TODO: We should attach the attributes defined in OMPKinds.def.
}

bool OMPInformationCache::declMatchesRTFTypes(
    Function *F, Type *RTFRetType, SmallVector<Type *, 8> &RTFArgTypes) {
  // TODO: We should output information to the user (under debug output
  //       and via remarks).

  if (!F)
    return false;
  if (F->getReturnType() != RTFRetType)
    return false;
  if (F->arg_size() != RTFArgTypes.size())
    return false;

  auto RTFTyIt = RTFArgTypes.begin();
  for (Argument &Arg : F->args()) {
    if (Arg.getType() != *RTFTyIt)
      return false;

    ++RTFTyIt;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Definitions of the MemoryTransfer helper structure.
//===----------------------------------------------------------------------===//

bool MemoryTransfer::getValuesInOffloadArrays() {
  // A runtime call that involves memory offloading looks something like:
  // call void @__tgt_target_data_begin(arg0, arg1,
  //   i8** %offload_baseptrs, i8** %offload_ptrs, i64* %offload_sizes,
  // ...)
  // So, the idea is to access the allocas that allocate space for these offload
  // arrays, offload_baseptrs, offload_ptrs, offload_sizes.
  // Therefore:
  // i8** %offload_baseptrs.
  Use *BasePtrsArg = RuntimeCall->arg_begin() + BasePtrsArgNum;
  // i8** %offload_ptrs.
  Use *PtrsArg = RuntimeCall->arg_begin() + PtrsArgNum;
  // i8** %offload_sizes.
  Use *SizesArg = RuntimeCall->arg_begin() + SizesArgNum;

  const DataLayout &DL = InfoCache.getDL();

  // Get values stored in **offload_baseptrs.
  auto *V = GetUnderlyingObject(BasePtrsArg->get(), DL);
  if (!isa<AllocaInst>(V)) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_baseptrs, only"
                      << " alloca arrays supported. In call to "
                      << RuntimeCall->getName() << " in function "
                      << RuntimeCall->getCaller()->getName() << "\n");
    return false;
  }

  auto *Array = cast<AllocaInst>(V);
  BasePtrs = OffloadArray::initialize(*Array, *RuntimeCall, InfoCache);
  if (!BasePtrs) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_baseptrs in call to "
                      << RuntimeCall->getName() << " in function "
                      << RuntimeCall->getCaller()->getName() << "\n");
    return false;
  }

  // Get values stored in **offload_ptrs.
  V = GetUnderlyingObject(PtrsArg->get(), DL);
  if (!isa<AllocaInst>(V)) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_ptrs, only"
                      << " alloca arrays supported. In call to "
                      << RuntimeCall->getName() << " in function "
                      << RuntimeCall->getCaller()->getName() << "\n");
    return false;
  }
  Array = cast<AllocaInst>(V);
  Ptrs = OffloadArray::initialize(*Array, *RuntimeCall, InfoCache);
  if (!Ptrs) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_ptrs in call to "
                      << RuntimeCall->getName() << " in function "
                      << RuntimeCall->getCaller()->getName() << "\n");
    return false;
  }

  // Get values stored in **offload_sizes.
  V = GetUnderlyingObject(SizesArg->get(), DL);
  // Sometimes the frontend generates this array as a constant global array.
  if (!isa<GlobalValue>(V)) {
    if (!isa<AllocaInst>(V)) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_sizes, only"
                        << " alloca arrays supported. In call to "
                        << RuntimeCall->getName() << " in function "
                        << RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }

    Array = cast<AllocaInst>(V);
    Sizes = OffloadArray::initialize(*Array, *RuntimeCall, InfoCache);
    if (!Sizes) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_sizes in call to "
                        << RuntimeCall->getName() << " in function "
                        << RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }
  }

  return true;
}

bool MemoryTransfer::detectIssue() {
  assert(BasePtrs && Ptrs && "No offload arrays to look at!");

  bool Success = getSetupInstructions(BasePtrs);
  if (!Success) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get setup instructions of "
                      << "offload_baseptrs. In call to "
                      << RuntimeCall->getName() << " in function "
                      << RuntimeCall->getCaller()->getName() << "\n");
    return false;
  }
  auto *BasePtrsGEP =
      cast<Instruction>(RuntimeCall->getArgOperand(BasePtrsArgNum));
  Issue.insert(BasePtrsGEP);

  Success = getSetupInstructions(Ptrs);
  if (!Success) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get setup instructions of "
                      << "offload_ptrs. In call to "
                      << RuntimeCall->getName() << " in function "
                      << RuntimeCall->getCaller()->getName() << "\n");
    return false;
  }
  auto *PtrsGEP =
      cast<Instruction>(RuntimeCall->getArgOperand(PtrsArgNum));
  Issue.insert(PtrsGEP);

  if (Sizes) {
    Success = getSetupInstructions(Sizes);
    if (!Success) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't get setup instructions of "
                        << "offload_sizes. In call to "
                        << RuntimeCall->getName() << " in function "
                        << RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }
    auto *SizesGEP =
        cast<Instruction>(RuntimeCall->getArgOperand(SizesArgNum));
    Issue.insert(SizesGEP);
  }

  return true;
}

bool MemoryTransfer::getSetupInstructions(std::unique_ptr<OffloadArray> &OA) {
  for (auto *S : OA->LastAccesses) {
    if (!getValueSetupInstructions(S))
      return false;

    if (!getPointerSetupInstructions(S))
      return false;

    Issue.insert(S);
  }
  return true;
}

bool MemoryTransfer::getPointerSetupInstructions(StoreInst *S) {
  auto *P = S->getPointerOperand();

  // TODO: P might be a global value. Make it general.
  if (!isa<Instruction>(P))
    return false;

  auto *DstInst = cast<Instruction>(P);
  if (isa<GetElementPtrInst>(DstInst)) {
    Issue.insert(DstInst);

  } else if (DstInst->isCast()) {
    auto *Casted = DstInst->getOperand(0);

    // TODO: Casted might be a global value. Make it general.
    if (!isa<Instruction>(Casted))
      return false;

    if (auto *GEP = dyn_cast<GetElementPtrInst>(Casted))
      Issue.insert(GEP);

    Issue.insert(DstInst);
  }

  return true;
}

bool MemoryTransfer::getValueSetupInstructions(StoreInst *S) {
  auto *V = S->getValueOperand();
  // Auxiliary storage to later insert the found instructions in the order
  // needed.
  SmallVector<Instruction *, 8> TempStorage;
  bool Success = false;
  unsigned MaxLookup = 6;
  for (unsigned I = 0; I < MaxLookup; ++I) {
    if (isa<AllocaInst>(V) || isa<Argument>(V) || isa<GlobalValue>(V) ||
        isa<Constant>(V)) {
      Success = true;
      break;
    }

    if (!isa<Instruction>(V)) {
      Success = false;
      break;
    }

    auto *Inst = cast<Instruction>(V);
    TempStorage.push_back(Inst);

    // FIXME: Inst might depend on more instructions through its second operand.
    V = Inst->getOperand(0);
  }

  if (Success)
    while (!TempStorage.empty())
      Issue.insert(TempStorage.pop_back_val());

  return Success;
}

bool MemoryTransfer::mayBeModifiedBy(Instruction *I) {
  assert(BasePtrs && Ptrs && "No offload addresses to analyze!");
  if (Issue.count(I))
    return false;

  if (mayModify(I, BasePtrs->StoredValues))
    return true;
  if (mayModify(I, Ptrs->StoredValues))
    return true;
  if (Sizes) {
    if (mayModify(I, Sizes->StoredValues))
      return true;
  }

  return false;
}

bool MemoryTransfer::mayModify(Instruction *I,
                               SmallVectorImpl<Value *> &Values) {
  assert(I && "Can't analyze nullptr!");
  auto *AAResults = InfoCache.getAnalysisResultForFunction<AAManager>(
      *RuntimeCall->getCaller());
  if (!AAResults) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get AAManager in function "
                      << RuntimeCall->getCaller()->getName() << "\n");
    return true;
  }

  const DataLayout &DL = InfoCache.getDL();

  if (isa<StoreInst>(I)) {
    auto *Dst = GetUnderlyingObject(I->getOperand(1), DL);
    for (auto *V : Values) {
      if (Dst == V) {
        return true;
      }
    }
  } else if (isa<CallInst>(I)) {
    for (auto *V : Values) {
      // FIXME: This usage of the AAResults is not working properly. It always
      //        returns that the call instruction I may modify a value V.
      //        For example:
      //        define i32 @func(double* noalias %a) {
      //        ...
      //        %1 = call i32 @rand()
      //        ...
      //        }
      //        The getModRefInfo always returns that rand() modifies %a, even
      //        though it has the noalias attribute.
      auto ModRefResult = AAResults->getModRefInfo(
          I, MemoryLocation(V, LocationSize::precise(
                                   V->getType()->getPrimitiveSizeInBits()))
          );
      if (isModSet(ModRefResult))
        return true;
    }
  }

  return true;
}

bool MemoryTransfer::split(Instruction *After, Instruction *Before) {
  assert((After || Before) &&
         "Must have a place to move the split runtime call");

  auto *M = RuntimeCall->getModule();
  auto &IRBuilder = InfoCache.OMPBuilder;
  // Add "issue" runtime call declaration.
  // declare %struct.tgt_async_info @__tgt_target_data_begin_issue(i64, i32,
  //   i8**, i8**, i64*, i64*)
  FunctionCallee IssueDecl = IRBuilder.getOrCreateRuntimeFunction(
      *M, OMPRTL___tgt_target_data_begin_issue);

  // Change RuntimeCall callsite for its asynchronous version.
  std::vector<Value *> Args;
  Args.reserve(RuntimeCall->getNumArgOperands());
  for (auto &Arg : RuntimeCall->args())
    Args.push_back(Arg.get());

  CallInst *IssueCallsite = CallInst::Create(
      IssueDecl, ArrayRef<Value *>(Args), "handle", RuntimeCall);
  RuntimeCall->removeFromParent();
  RuntimeCall->deleteValue();
  Issue.insert(IssueCallsite);

  // Add "wait" runtime call declaration.
  // declare void @__tgt_target_data_begin_wait(i64, %struct.__tgt_async_info)
  FunctionCallee WaitDecl = IRBuilder.getOrCreateRuntimeFunction(
      *M, OMPRTL___tgt_target_data_begin_wait);

  // Add "wait" call site.
  const unsigned WaitNumParams = 2;
  Value *WaitParams[] = {
      IssueCallsite->getArgOperand(0), // device_id.
      IssueCallsite // returned handle.
  };
  CallInst *WaitCallsite = CallInst::Create(
      WaitDecl, ArrayRef<Value*>(WaitParams, WaitNumParams), /*NameStr=*/"",
          /*InsertBefore=*/(Instruction *)nullptr);

  // Move wait.
  if (!Before)
    WaitCallsite->insertAfter(IssueCallsite);
  else
    WaitCallsite->insertBefore(Before);

  if (After)
    moveIssue(After);

  return true;
}

void MemoryTransfer::moveIssue(Instruction *After) {
  Instruction *Before = After->getNextNode();
  for (auto *I : Issue) {
    I->removeFromParent();
    I->insertBefore(Before);
  }
}

std::unique_ptr<OffloadArray> OffloadArray::initialize(
    AllocaInst &Array, Instruction &Before, InformationCache &InfoCache) {
  if (!Array.getAllocatedType()->isArrayTy()) {
    LLVM_DEBUG(dbgs() << TAG << "Allocated type is not array.\n");
    return nullptr;
  }

  auto OA = std::make_unique<OffloadArray>(Array, InfoCache);
  bool Success = OA->getValues(Before);
  if (!Success) {
    LLVM_DEBUG(dbgs() << TAG << "Error getting values in offload array.\n");
    return nullptr;
  }

  return OA;
}

bool OffloadArray::getValues(Instruction &Before) {
  // Initialize container.
  const uint64_t NumValues =
      Array.getAllocatedType()->getArrayNumElements();
  StoredValues.assign(NumValues, nullptr);
  LastAccesses.assign(NumValues, nullptr);

  // TODO: This assumes the instruction \p Before is in the same BasicBlock
  //       as OffloadArray::Array. Make it general, for any control flow graph.
  auto *BB = Array.getParent();
  if (BB != Before.getParent()) {
    LLVM_DEBUG(dbgs() << TAG << "The lower limit instruction is in a"
                      << " different BasicBlock.\n");
    return false;
  }

  const DataLayout &DL = InfoCache.getDL();
  for (auto &I : *BB) {
    if (&I == &Before) break;

    if (auto *S = dyn_cast<StoreInst>(&I)) {
      auto *Dst = GetUnderlyingObject(S->getPointerOperand(), DL);

      if (Dst == &Array) {
        int32_t Idx = getAccessedIdx(*S);
        if (Idx < 0) {
          LLVM_DEBUG(dbgs() << TAG << "Unexpected StoreInst\n");
          return false;
        }

        StoredValues[Idx] = GetUnderlyingObject(S->getValueOperand(), DL);
        LastAccesses[Idx] = S;
      }
    }
  }

  return isFilled();
}

int32_t OffloadArray::getAccessedIdx(StoreInst &S) {
  auto *Dst = S.getOperand(1);
  if (!isa<Instruction>(Dst)) {
    LLVM_DEBUG(dbgs() << TAG << "Unrecognized store pattern.\n");
    return -1;
  }
  auto *DstInst = cast<Instruction>(Dst);

  Value *Access = DstInst;
  if (DstInst->isCast()) {
    Access = DstInst->getOperand(0);

    // Direct cast from the AllocaInst, which means a store to the
    // first position of the array.
    if (Access == &Array) return 0;
  }

  if (!isa<GetElementPtrInst>(Access)) {
    LLVM_DEBUG(dbgs() << TAG << "Unrecognized store pattern.\n");
    return -1;
  }
  auto *GEPInst = cast<GetElementPtrInst>(Access);

  auto *ArrayIdx = GEPInst->idx_begin() + 1;
  if (ArrayIdx == GEPInst->idx_end()) {
    LLVM_DEBUG(dbgs() << TAG << "Unrecognized store pattern.\n");
    return -1;
  }

  return OpenMPOpt::getIntLiteral(ArrayIdx->get());
}

bool OffloadArray::isFilled() {
  const unsigned NumValues = StoredValues.size();
  for (unsigned I = 0; I < NumValues; ++I) {
    if (!StoredValues[I] || !LastAccesses[I])
      return false;
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Declarations and definitions of AAICVTracker.
//===----------------------------------------------------------------------===//
namespace {

/// Abstract Attribute for tracking ICV values.
struct AAICVTracker : public StateWrapper<BooleanState, AbstractAttribute> {
  using Base = StateWrapper<BooleanState, AbstractAttribute>;
  AAICVTracker(const IRPosition &IRP, Attributor &A) : Base(IRP) {}

  /// Returns true if value is assumed to be tracked.
  bool isAssumedTracked() const { return getAssumed(); }

  /// Returns true if value is known to be tracked.
  bool isKnownTracked() const { return getAssumed(); }

  /// Create an abstract attribute biew for the position \p IRP.
  static AAICVTracker &createForPosition(const IRPosition &IRP, Attributor &A);

  /// Return the value with which \p I can be replaced for specific \p ICV.
  virtual Value *getReplacementValue(InternalControlVar ICV,
                                     const Instruction *I, Attributor &A) = 0;

  /// See AbstractAttribute::getName()
  const std::string getName() const override { return "AAICVTracker"; }

  static const char ID;
};

struct AAICVTrackerFunction : public AAICVTracker {
  AAICVTrackerFunction(const IRPosition &IRP, Attributor &A)
      : AAICVTracker(IRP, A) {}

  // FIXME: come up with better string.
  const std::string getAsStr() const override { return "ICVTracker"; }

  // FIXME: come up with some stats.
  void trackStatistics() const override {}

  /// TODO: decide whether to deduplicate here, or use current
  /// deduplicateRuntimeCalls function.
  ChangeStatus manifest(Attributor &A) override {
    ChangeStatus Changed = ChangeStatus::UNCHANGED;

    for (InternalControlVar &ICV : TrackableICVs)
      if (deduplicateICVGetters(ICV, A))
        Changed = ChangeStatus::CHANGED;

    return Changed;
  }

  bool deduplicateICVGetters(InternalControlVar &ICV, Attributor &A) {
    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    auto &ICVInfo = OMPInfoCache.ICVs[ICV];
    auto &GetterRFI = OMPInfoCache.RFIs[ICVInfo.Getter];

    bool Changed = false;

    auto ReplaceAndDeleteCB = [&](Use &U, Function &Caller) {
      CallInst *CI = OpenMPOpt::getCallIfRegularCall(U, &GetterRFI);
      Instruction *UserI = cast<Instruction>(U.getUser());
      Value *ReplVal = getReplacementValue(ICV, UserI, A);

      if (!ReplVal || !CI)
        return false;

      A.removeCallSite(CI);
      CI->replaceAllUsesWith(ReplVal);
      CI->eraseFromParent();
      Changed = true;
      return true;
    };

    GetterRFI.foreachUse(ReplaceAndDeleteCB);
    return Changed;
  }

  // Map of ICV to their values at specific program point.
  EnumeratedArray<SmallSetVector<ICVValue, 4>, InternalControlVar,
      InternalControlVar::ICV___last>
      ICVValuesMap;

  // Currently only nthreads is being tracked.
  // this array will only grow with time.
  InternalControlVar TrackableICVs[1] = {ICV_nthreads};

  ChangeStatus updateImpl(Attributor &A) override {
    ChangeStatus HasChanged = ChangeStatus::UNCHANGED;

    Function *F = getAnchorScope();

    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());

    for (InternalControlVar ICV : TrackableICVs) {
      auto &SetterRFI = OMPInfoCache.RFIs[OMPInfoCache.ICVs[ICV].Setter];

      auto TrackValues = [&](Use &U, Function &) {
        CallInst *CI = OpenMPOpt::getCallIfRegularCall(U);
        if (!CI)
          return false;

        // FIXME: handle setters with more that 1 arguments.
        /// Track new value.
        if (ICVValuesMap[ICV].insert(ICVValue(CI, CI->getArgOperand(0))))
          HasChanged = ChangeStatus::CHANGED;

        return false;
      };

      SetterRFI.foreachUse(TrackValues, F);
    }

    return HasChanged;
  }

  /// Return the value with which \p I can be replaced for specific \p ICV.
  Value *getReplacementValue(InternalControlVar ICV, const Instruction *I,
                             Attributor &A) override {
    const BasicBlock *CurrBB = I->getParent();

    auto &ValuesSet = ICVValuesMap[ICV];
    auto &OMPInfoCache = static_cast<OMPInformationCache &>(A.getInfoCache());
    auto &GetterRFI = OMPInfoCache.RFIs[OMPInfoCache.ICVs[ICV].Getter];

    for (const auto &ICVVal : ValuesSet) {
      if (CurrBB == ICVVal.Inst->getParent()) {
        if (!ICVVal.Inst->comesBefore(I))
          continue;

        // both instructions are in the same BB and at \p I we know the ICV
        // value.
        while (I != ICVVal.Inst) {
          // we don't yet know if a call might update an ICV.
          // TODO: check callsite AA for value.
          if (const auto *CB = dyn_cast<CallBase>(I))
            if (CB->getCalledFunction() != GetterRFI.Declaration)
              return nullptr;

          I = I->getPrevNode();
        }

        // No call in between, return the value.
        return ICVVal.TrackedValue;
      }
    }

    // No value was tracked.
    return nullptr;
  }
};
} // namespace

const char AAICVTracker::ID = 0;

AAICVTracker &AAICVTracker::createForPosition(const IRPosition &IRP,
                                              Attributor &A) {
  AAICVTracker *AA = nullptr;
  switch (IRP.getPositionKind()) {
  case IRPosition::IRP_INVALID:
  case IRPosition::IRP_FLOAT:
  case IRPosition::IRP_ARGUMENT:
  case IRPosition::IRP_RETURNED:
  case IRPosition::IRP_CALL_SITE_RETURNED:
  case IRPosition::IRP_CALL_SITE_ARGUMENT:
  case IRPosition::IRP_CALL_SITE:
    llvm_unreachable("ICVTracker can only be created for function position!");
  case IRPosition::IRP_FUNCTION:
    AA = new (A.Allocator) AAICVTrackerFunction(IRP, A);
    break;
  }

  return *AA;
}

//===----------------------------------------------------------------------===//
// Definitions of the OpenMPOpt structure.
//===----------------------------------------------------------------------===//

bool OpenMPOpt::run() {
  bool Changed = false;

  LLVM_DEBUG(dbgs() << TAG << "Run on SCC with " << SCC.size()
                    << " functions in a slice with "
                    << OMPInfoCache.ModuleSlice.size() << " functions\n");

  /// Print initial ICV values for testing.
  /// FIXME: This should be done from the Attributor once it is added.
  if (PrintICVValues) {
    InternalControlVar ICVs[] = {ICV_nthreads, ICV_active_levels, ICV_cancel};

    for (Function *F : OMPInfoCache.ModuleSlice) {
      for (auto ICV : ICVs) {
        auto ICVInfo = OMPInfoCache.ICVs[ICV];
        auto Remark = [&](OptimizationRemark OR) {
          return OR << "OpenMP ICV " << ore::NV("OpenMPICV", ICVInfo.Name)
                    << " Value: "
                    << (ICVInfo.InitValue
                        ? ICVInfo.InitValue->getValue().toString(10, true)
                        : "IMPLEMENTATION_DEFINED");
        };

        emitRemarkOnFunction(F, "OpenMPICVTracker", Remark);
      }
    }
  }

  Changed |= runAttributor();
  Changed |= deduplicateRuntimeCalls();
  Changed |= deleteParallelRegions();
  if (SplitMemoryTransfers)
    Changed |= hideMemTransfersLatency();


  return Changed;
}

CallInst *OpenMPOpt::getCallIfRegularCall(
    Use &U, OMPInformationCache::RuntimeFunctionInfo *RFI) {
  CallInst *CI = dyn_cast<CallInst>(U.getUser());
  if (CI && CI->isCallee(&U) && !CI->hasOperandBundles() &&
      (!RFI || CI->getCalledFunction() == RFI->Declaration))
    return CI;
  return nullptr;
}

CallInst *OpenMPOpt::getCallIfRegularCall(
    Value &V, OMPInformationCache::RuntimeFunctionInfo *RFI) {
  CallInst *CI = dyn_cast<CallInst>(&V);
  if (CI && !CI->hasOperandBundles() &&
      (!RFI || CI->getCalledFunction() == RFI->Declaration))
    return CI;
  return nullptr;
}

bool OpenMPOpt::deleteParallelRegions() {
  const unsigned CallbackCalleeOperand = 2;

  OMPInformationCache::RuntimeFunctionInfo &RFI =
      OMPInfoCache.RFIs[OMPRTL___kmpc_fork_call];

  if (!RFI.Declaration)
    return false;

  bool Changed = false;
  auto DeleteCallCB = [&](Use &U, Function &) {
    CallInst *CI = getCallIfRegularCall(U);
    if (!CI)
      return false;
    auto *Fn = dyn_cast<Function>(
        CI->getArgOperand(CallbackCalleeOperand)->stripPointerCasts());
    if (!Fn)
      return false;
    if (!Fn->onlyReadsMemory())
      return false;
    if (!Fn->hasFnAttribute(Attribute::WillReturn))
      return false;

    LLVM_DEBUG(dbgs() << TAG << "Delete read-only parallel region in "
                      << CI->getCaller()->getName() << "\n");

    auto Remark = [&](OptimizationRemark OR) {
      return OR << "Parallel region in "
                << ore::NV("OpenMPParallelDelete", CI->getCaller()->getName())
                << " deleted";
    };
    emitRemark<OptimizationRemark>(CI, "OpenMPParallelRegionDeletion",
                                   Remark);

    CGUpdater.removeCallSite(*CI);
    CI->eraseFromParent();
    Changed = true;
    ++NumOpenMPParallelRegionsDeleted;
    return true;
  };

  RFI.foreachUse(DeleteCallCB);

  return Changed;
}

bool OpenMPOpt::deduplicateRuntimeCalls() {
  bool Changed = false;

  RuntimeFunction DeduplicableRuntimeCallIDs[] = {
      OMPRTL_omp_get_num_threads,
      OMPRTL_omp_in_parallel,
      OMPRTL_omp_get_cancellation,
      OMPRTL_omp_get_thread_limit,
      OMPRTL_omp_get_supported_active_levels,
      OMPRTL_omp_get_level,
      OMPRTL_omp_get_ancestor_thread_num,
      OMPRTL_omp_get_team_size,
      OMPRTL_omp_get_active_level,
      OMPRTL_omp_in_final,
      OMPRTL_omp_get_proc_bind,
      OMPRTL_omp_get_num_places,
      OMPRTL_omp_get_num_procs,
      OMPRTL_omp_get_place_num,
      OMPRTL_omp_get_partition_num_places,
      OMPRTL_omp_get_partition_place_nums};

  // Global-tid is handled separately.
  SmallSetVector<Value *, 16> GTIdArgs;
  collectGlobalThreadIdArguments(GTIdArgs);
  LLVM_DEBUG(dbgs() << TAG << "Found " << GTIdArgs.size()
                    << " global thread ID arguments\n");

  for (Function *F : SCC) {
    for (auto DeduplicableRuntimeCallID : DeduplicableRuntimeCallIDs)
      deduplicateRuntimeCalls(*F,
                              OMPInfoCache.RFIs[DeduplicableRuntimeCallID]);

    // __kmpc_global_thread_num is special as we can replace it with an
    // argument in enough cases to make it worth trying.
    Value *GTIdArg = nullptr;
    for (Argument &Arg : F->args())
      if (GTIdArgs.count(&Arg)) {
        GTIdArg = &Arg;
        break;
      }
    Changed |= deduplicateRuntimeCalls(
        *F, OMPInfoCache.RFIs[OMPRTL___kmpc_global_thread_num], GTIdArg);
  }

  return Changed;
}

bool OpenMPOpt::hideMemTransfersLatency() {
  OMPInformationCache::RuntimeFunctionInfo &RFI =
      OMPInfoCache.RFIs[OMPRTL___tgt_target_data_begin];

  bool Changed = false;
  auto SplitDataTransfer = [&] (Use &U, Function &Decl) {
    auto *RTCall = getCallIfRegularCall(U, &RFI);
    if (!RTCall)
      return false;

    MemoryTransfer MT(RTCall, OMPInfoCache);
    bool Success = MT.getValuesInOffloadArrays();
    if (!Success) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload arrays in call to "
                        << MT.RuntimeCall->getName() << " in function "
                        << MT.RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }

    Success = MT.detectIssue();
    if (!Success) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't detect issue in call to "
                        << MT.RuntimeCall->getName() << " in function "
                        << MT.RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }

    auto *After = canBeMovedUpwards(MT);
    auto *Before = canBeMovedDownwards(MT);
    return (After || Before) && MT.split(After, Before);
  };

  RFI.foreachUse(SplitDataTransfer);
  return Changed;
}

Instruction *OpenMPOpt::canBeMovedUpwards(MemoryTransfer &MT) {
  assert(MT.Issue.size() > 0 && "There's not set of instructions to be moved!");

  CallInst *RC = MT.RuntimeCall;
  auto *MSSAResult =
      OMPInfoCache.getAnalysisResultForFunction<MemorySSAAnalysis>(
          *RC->getCaller());
  if (!MSSAResult) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get MemorySSAAnalysis in function "
                      << RC->getCaller()->getName() << "\n");
    return nullptr;
  }

  auto &MSSA = MSSAResult->getMSSA();
  auto *MSSAWalker = MSSA.getWalker();
  const auto *LiveOnEntry = MSSA.getLiveOnEntryDef();
  auto *MemAccess = MSSAWalker->getClobberingMemoryAccess(RC);

  while (MemAccess != LiveOnEntry) {
    if (!isa<MemoryDef>(MemAccess))
      continue;

    auto *MemInst = (cast<MemoryDef>(MemAccess))->getMemoryInst();
    if (MT.mayBeModifiedBy(MemInst)) {
      // If MemInst is not the instruction immediately before the Issue.
      if (!MT.Issue.count(MemInst->getNextNode()))
        return MemInst;

      return nullptr;
    }

    MemAccess = MSSAWalker->getClobberingMemoryAccess(MemAccess);
  }

  return nullptr;
}

Instruction *OpenMPOpt::canBeMovedDownwards(MemoryTransfer &MT) {
  assert(MT.Issue.size() > 0 && "There's not set of instructions to be moved!");

  // FIXME: This traverses only the BasicBlock where MT is. Make it traverse
  //        the CFG.
  GlobalValue *TgtTargetDecl = M.getNamedValue("__tgt_target");
  GlobalValue *TgtTargetTeamsDecl = M.getNamedValue("__tgt_target_teams");
  GlobalValue *TgtTargetDataEndDecl = M.getNamedValue("__tgt_target_data_end");
  CallInst *RC = MT.RuntimeCall;
  auto *I = RC->getNextNode();
  while (I) {
    if (auto *C = dyn_cast<CallInst>(I)) {
      auto *Callee = C->getCalledFunction();
      if (Callee == TgtTargetDecl)
        return I;
      if (Callee == TgtTargetTeamsDecl)
        return I;
      if (Callee == TgtTargetDataEndDecl)
        return I;
    }

    I = I->getNextNode();
  }

  // Return end of BasicBlock.
  return &*(RC->getParent()->end());
}

Value *OpenMPOpt::combinedIdentStruct(Value *CurrentIdent, Value *NextIdent,
    bool GlobalOnly, bool &SingleChoice) {
  if (CurrentIdent == NextIdent)
    return CurrentIdent;

  // TODO: Figure out how to actually combine multiple debug locations. For
  //       now we just keep an existing one if there is a single choice.
  if (!GlobalOnly || isa<GlobalValue>(NextIdent)) {
    SingleChoice = !CurrentIdent;
    return NextIdent;
  }
  return nullptr;
}

Value * OpenMPOpt::getCombinedIdentFromCallUsesIn(
    OMPInformationCache::RuntimeFunctionInfo &RFI,
    Function &F, bool GlobalOnly) {
  bool SingleChoice = true;
  Value *Ident = nullptr;
  auto CombineIdentStruct = [&](Use &U, Function &Caller) {
    CallInst *CI = getCallIfRegularCall(U, &RFI);
    if (!CI || &F != &Caller)
      return false;
    Ident = combinedIdentStruct(Ident, CI->getArgOperand(0),
        /* GlobalOnly */ true, SingleChoice);
    return false;
  };
  RFI.foreachUse(CombineIdentStruct);

  if (!Ident || !SingleChoice) {
    // The IRBuilder uses the insertion block to get to the module, this is
    // unfortunate but we work around it for now.
    if (!OMPInfoCache.OMPBuilder.getInsertionPoint().getBlock())
      OMPInfoCache.OMPBuilder.updateToLocation(OpenMPIRBuilder::InsertPointTy(
          &F.getEntryBlock(), F.getEntryBlock().begin()));
    // Create a fallback location if non was found.
    // TODO: Use the debug locations of the calls instead.
    Constant *Loc = OMPInfoCache.OMPBuilder.getOrCreateDefaultSrcLocStr();
    Ident = OMPInfoCache.OMPBuilder.getOrCreateIdent(Loc);
  }
  return Ident;
}

bool OpenMPOpt::deduplicateRuntimeCalls(
    Function &F, OMPInformationCache::RuntimeFunctionInfo &RFI,
    Value *ReplVal) {
  auto *UV = RFI.getUseVector(F);
  if (!UV || UV->size() + (ReplVal != nullptr) < 2)
    return false;

  LLVM_DEBUG(
      dbgs() << TAG << "Deduplicate " << UV->size() << " uses of " << RFI.Name
             << (ReplVal ? " with an existing value\n" : "\n") << "\n");

  assert((!ReplVal || (isa<Argument>(ReplVal) &&
                       cast<Argument>(ReplVal)->getParent() == &F)) &&
         "Unexpected replacement value!");

  // TODO: Use dominance to find a good position instead.
  auto CanBeMoved = [](CallBase &CB) {
    unsigned NumArgs = CB.getNumArgOperands();
    if (NumArgs == 0)
      return true;
    if (CB.getArgOperand(0)->getType() != IdentPtr)
      return false;
    for (unsigned u = 1; u < NumArgs; ++u)
      if (isa<Instruction>(CB.getArgOperand(u)))
        return false;
    return true;
  };

  if (!ReplVal) {
    for (Use *U : *UV)
      if (CallInst *CI = getCallIfRegularCall(*U, &RFI)) {
        if (!CanBeMoved(*CI))
          continue;

        auto Remark = [&](OptimizationRemark OR) {
          auto newLoc = &*F.getEntryBlock().getFirstInsertionPt();
          return OR << "OpenMP runtime call "
                    << ore::NV("OpenMPOptRuntime", RFI.Name) << " moved to "
                    << ore::NV("OpenMPRuntimeMoves", newLoc->getDebugLoc());
        };
        emitRemark<OptimizationRemark>(CI, "OpenMPRuntimeCodeMotion", Remark);

        CI->moveBefore(&*F.getEntryBlock().getFirstInsertionPt());
        ReplVal = CI;
        break;
      }
    if (!ReplVal)
      return false;
  }

  // If we use a call as a replacement value we need to make sure the ident is
  // valid at the new location. For now we just pick a global one, either
  // existing and used by one of the calls, or created from scratch.
  if (CallBase *CI = dyn_cast<CallBase>(ReplVal)) {
    if (CI->getNumArgOperands() > 0 &&
        CI->getArgOperand(0)->getType() == IdentPtr) {
      Value *Ident = getCombinedIdentFromCallUsesIn(RFI, F,
          /* GlobalOnly */ true);
      CI->setArgOperand(0, Ident);
    }
  }

  bool Changed = false;
  auto ReplaceAndDeleteCB = [&](Use &U, Function &Caller) {
    CallInst *CI = getCallIfRegularCall(U, &RFI);
    if (!CI || CI == ReplVal || &F != &Caller)
      return false;
    assert(CI->getCaller() == &F && "Unexpected call!");

    auto Remark = [&](OptimizationRemark OR) {
      return OR << "OpenMP runtime call "
                << ore::NV("OpenMPOptRuntime", RFI.Name) << " deduplicated";
    };
    emitRemark<OptimizationRemark>(CI, "OpenMPRuntimeDeduplicated", Remark);

    CGUpdater.removeCallSite(*CI);
    CI->replaceAllUsesWith(ReplVal);
    CI->eraseFromParent();
    ++NumOpenMPRuntimeCallsDeduplicated;
    Changed = true;
    return true;
  };
  RFI.foreachUse(ReplaceAndDeleteCB);

  return Changed;
}

void OpenMPOpt::collectGlobalThreadIdArguments(
    SmallSetVector<Value *, 16> &GTIdArgs) {
  // TODO: Below we basically perform a fixpoint iteration with a pessimistic
  //       initialization. We could define an AbstractAttribute instead and
  //       run the Attributor here once it can be run as an SCC pass.

  // Helper to check the argument \p ArgNo at all call sites of \p F for
  // a GTId.
  auto CallArgOpIsGTId = [&](Function &F, unsigned ArgNo, CallInst &RefCI) {
    if (!F.hasLocalLinkage())
      return false;
    for (Use &U : F.uses()) {
      if (CallInst *CI = getCallIfRegularCall(U)) {
        Value *ArgOp = CI->getArgOperand(ArgNo);
        if (CI == &RefCI || GTIdArgs.count(ArgOp) ||
            getCallIfRegularCall(
                *ArgOp, &OMPInfoCache.RFIs[OMPRTL___kmpc_global_thread_num]))
          continue;
      }
      return false;
    }
    return true;
  };

  // Helper to identify uses of a GTId as GTId arguments.
  auto AddUserArgs = [&](Value &GTId) {
    for (Use &U : GTId.uses())
      if (CallInst *CI = dyn_cast<CallInst>(U.getUser()))
        if (CI->isArgOperand(&U))
          if (Function *Callee = CI->getCalledFunction())
            if (CallArgOpIsGTId(*Callee, U.getOperandNo(), *CI))
              GTIdArgs.insert(Callee->getArg(U.getOperandNo()));
  };

  // The argument users of __kmpc_global_thread_num calls are GTIds.
  OMPInformationCache::RuntimeFunctionInfo &GlobThreadNumRFI =
      OMPInfoCache.RFIs[OMPRTL___kmpc_global_thread_num];

  GlobThreadNumRFI.foreachUse([&](Use &U, Function &F) {
    if (CallInst *CI =
        getCallIfRegularCall(U, &GlobThreadNumRFI))
      AddUserArgs(*CI);
    return false;
  });

  // Transitively search for more arguments by looking at the users of the
  // ones we know already. During the search the GTIdArgs vector is extended
  // so we cannot cache the size nor can we use a range based for.
  for (unsigned u = 0; u < GTIdArgs.size(); ++u)
    AddUserArgs(*GTIdArgs[u]);
}

template <typename RemarkKind, typename RemarkCallBack>
void OpenMPOpt::emitRemark(Instruction *Inst, StringRef RemarkName,
                           RemarkCallBack &&RemarkCB) {
  Function *F = Inst->getParent()->getParent();
  auto &ORE = OREGetter(F);

  ORE.emit(
      [&]() { return RemarkCB(RemarkKind(DEBUG_TYPE, RemarkName, Inst)); });
}

void OpenMPOpt::emitRemarkOnFunction(
    Function *F, StringRef RemarkName,
    function_ref<OptimizationRemark(OptimizationRemark &&)> &&RemarkCB) {
  auto &ORE = OREGetter(F);

  ORE.emit([&]() {
    return RemarkCB(OptimizationRemark(DEBUG_TYPE, RemarkName, F));
  });
}

bool OpenMPOpt::runAttributor() {
  if (SCC.empty())
    return false;

  registerAAs();

  ChangeStatus Changed = A.run();

  LLVM_DEBUG(dbgs() << "[Attributor] Done with " << SCC.size()
                    << " functions, result: " << Changed << ".\n");

  return Changed == ChangeStatus::CHANGED;
}

void OpenMPOpt::registerAAs() {
  for (Function *F : SCC) {
    if (F->isDeclaration())
      continue;

    A.getOrCreateAAFor<AAICVTracker>(IRPosition::function(*F));
  }
}

//===----------------------------------------------------------------------===//
// Definitions of the OpenMPOptPass.
//===----------------------------------------------------------------------===//

PreservedAnalyses OpenMPOptPass::run(LazyCallGraph::SCC &C,
                                     CGSCCAnalysisManager &AM,
                                     LazyCallGraph &CG, CGSCCUpdateResult &UR) {
  if (!containsOpenMP(*C.begin()->getFunction().getParent(), OMPInModule))
    return PreservedAnalyses::all();

  if (DisableOpenMPOptimizations)
    return PreservedAnalyses::all();

  SmallPtrSet<Function *, 16> ModuleSlice;
  SmallVector<Function *, 16> SCC;
  for (LazyCallGraph::Node &N : C) {
    SCC.push_back(&N.getFunction());
    ModuleSlice.insert(SCC.back());
  }

  if (SCC.empty())
    return PreservedAnalyses::all();

  FunctionAnalysisManager &FAM =
      AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

  AnalysisGetter AG(FAM);

  auto OREGetter = [&FAM](Function *F) -> OptimizationRemarkEmitter & {
    return FAM.getResult<OptimizationRemarkEmitterAnalysis>(*F);
  };

  CallGraphUpdater CGUpdater;
  CGUpdater.initialize(CG, C, AM, UR);

  SetVector<Function *> Functions(SCC.begin(), SCC.end());
  BumpPtrAllocator Allocator;
  OMPInformationCache InfoCache(*(Functions.back()->getParent()), AG, Allocator,
                                /*CGSCC*/ &Functions, ModuleSlice);

  Attributor A(Functions, InfoCache, CGUpdater);

  // TODO: Compute the module slice we are allowed to look at.
  OpenMPOpt OMPOpt(SCC, CGUpdater, OREGetter, InfoCache, A);
  bool Changed = OMPOpt.run();
  (void)Changed;
  return PreservedAnalyses::all();
}

namespace {

struct OpenMPOptLegacyPass : public CallGraphSCCPass {
  CallGraphUpdater CGUpdater;
  OpenMPInModule OMPInModule;
  static char ID;

  OpenMPOptLegacyPass() : CallGraphSCCPass(ID) {
    initializeOpenMPOptLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  bool doInitialization(CallGraph &CG) override {
    // Disable the pass if there is no OpenMP (runtime call) in the module.
    containsOpenMP(CG.getModule(), OMPInModule);
    return false;
  }

  bool runOnSCC(CallGraphSCC &CGSCC) override {
    if (!containsOpenMP(CGSCC.getCallGraph().getModule(), OMPInModule))
      return false;
    if (DisableOpenMPOptimizations || skipSCC(CGSCC))
      return false;

    SmallPtrSet<Function *, 16> ModuleSlice;
    SmallVector<Function *, 16> SCC;
    for (CallGraphNode *CGN : CGSCC)
      if (Function *Fn = CGN->getFunction())
        if (!Fn->isDeclaration()) {
          SCC.push_back(Fn);
          ModuleSlice.insert(Fn);
        }

    if (SCC.empty())
      return false;

    CallGraph &CG = getAnalysis<CallGraphWrapperPass>().getCallGraph();
    CGUpdater.initialize(CG, CGSCC);

    // Maintain a map of functions to avoid rebuilding the ORE
    DenseMap<Function *, std::unique_ptr<OptimizationRemarkEmitter>> OREMap;
    auto OREGetter = [&OREMap](Function *F) -> OptimizationRemarkEmitter & {
      std::unique_ptr<OptimizationRemarkEmitter> &ORE = OREMap[F];
      if (!ORE)
        ORE = std::make_unique<OptimizationRemarkEmitter>(F);
      return *ORE;
    };

    AnalysisGetter AG;
    SetVector<Function *> Functions(SCC.begin(), SCC.end());
    BumpPtrAllocator Allocator;
    OMPInformationCache InfoCache(*(Functions.back()->getParent()), AG,
                                  Allocator,
                                  /*CGSCC*/ &Functions, ModuleSlice);

    Attributor A(Functions, InfoCache, CGUpdater);

    // TODO: Compute the module slice we are allowed to look at.
    OpenMPOpt OMPOpt(SCC, CGUpdater, OREGetter, InfoCache, A);
    return OMPOpt.run();
  }

  bool doFinalization(CallGraph &CG) override { return CGUpdater.finalize(); }
};

} // end anonymous namespace

bool llvm::omp::containsOpenMP(Module &M, OpenMPInModule &OMPInModule) {
  if (OMPInModule.isKnown())
    return OMPInModule;

#define OMP_RTL(_Enum, _Name, ...)                                             \
  if (M.getFunction(_Name))                                                    \
    return OMPInModule = true;
#include "llvm/Frontend/OpenMP/OMPKinds.def"
  return OMPInModule = false;
}

char OpenMPOptLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(OpenMPOptLegacyPass, "openmpopt",
                      "OpenMP specific optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(OpenMPOptLegacyPass, "openmpopt",
                    "OpenMP specific optimizations", false, false)

Pass *llvm::createOpenMPOptLegacyPass() { return new OpenMPOptLegacyPass(); }
