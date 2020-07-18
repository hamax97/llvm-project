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

  auto *RTFTyIt = RTFArgTypes.begin();
  for (Argument &Arg : F->args()) {
    if (Arg.getType() != *RTFTyIt)
      return false;

    ++RTFTyIt;
  }

  return true;
}

bool OffloadArray::isFilled(const SmallVectorImpl<Value *> &V) {
  for (auto *E : V)
    if (!E)
      return false;

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

    auto *MSSAResult =
        OMPInfoCache.getAnalysisResultForFunction<MemorySSAAnalysis>(
            *RTCall->getCaller());
    if (!MSSAResult)
      return false;

    auto &MSSA = MSSAResult->getMSSA();
    MemoryTransfer MT(RTCall, MSSA);
    Changed = splitMemoryTransfer(MT);
    return Changed;
  };

  RFI.foreachUse(SplitDataTransfer);
  return Changed;
}

bool OpenMPOpt::splitMemoryTransfer(MemoryTransfer &MT) {
  bool Changed = false;
  bool Success = getValuesInOfflArrays(MT);
  if (!Success) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload arrays in call to "
                      << MT.RuntimeCall->getName() << " in function "
                      << MT.RuntimeCall->getCaller()->getName() << "\n");
    return false;
  }

  return Changed;
}

bool OpenMPOpt::getValuesInOfflArrays(MemoryTransfer &MT) {
  auto *RuntimeCall = MT.RuntimeCall;
  const unsigned BasePtrs = 2; // **offload_baseptrs.
  const unsigned Ptrs = 3; // **offload_ptrs.
  const unsigned Sizes = 4; // **offload_sizes.
  auto *BasePtrsArg = RuntimeCall->arg_begin() + BasePtrs;
  auto *PtrsArg = RuntimeCall->arg_begin() + Ptrs;
  auto *SizesArg = RuntimeCall->arg_begin() + Sizes;
  const auto &DL = OMPInfoCache.getDL();

  // Get values stored in **offload_baseptrs.
  auto *V = GetUnderlyingObject(BasePtrsArg->get(), DL);
  if (!isa<GlobalValue>(V)) {
    bool Success = getValuesInOfflArray(V, *MT.BasePtrs, RuntimeCall);
    if (!Success) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_baseptrs in call to "
                        << RuntimeCall->getName() << " in function "
                        << RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }
  }

  // Get values stored in **offload_ptrs.
  V = GetUnderlyingObject(PtrsArg->get(), DL);
  if (!isa<GlobalValue>(V)) {
    bool Success = getValuesInOfflArray(V, *MT.Ptrs, RuntimeCall);
    if (!Success) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_ptrs in call to "
                        << RuntimeCall->getName() << " in function "
                        << RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }
  }

  // Get values stored in **offload_sizes.
  V = GetUnderlyingObject(SizesArg->get(), DL);
  if (!isa<GlobalValue>(V)) {
    bool Success = getValuesInOfflArray(V, *MT.Sizes, RuntimeCall);
    if (!Success) {
      LLVM_DEBUG(dbgs() << TAG << "Couldn't get offload_sizes in call to "
                        << RuntimeCall->getName() << " in function "
                        << RuntimeCall->getCaller()->getName() << "\n");
      return false;
    }
  }
  return true;
}

bool OpenMPOpt::getValuesInOfflArray(
    Value *OfflArray, MemoryTransfer::OffloadArray &Dst, Instruction *Before) {
  assert(OfflArray && "Can't get values of nullptr!");

  if (!isa<AllocaInst>(OfflArray)) {
    LLVM_DEBUG(dbgs() << TAG << "Only alloca arrays supported.\n");
    return false;
  }

  auto *ArrayAlloc = cast<AllocaInst>(OfflArray);
  const uint64_t NumValues =
      ArrayAlloc->getAllocatedType()->getArrayNumElements();
  Dst.initialize(NumValues);

  return getLastAccessesToOfflArray(ArrayAlloc, Dst.LastAccesses, Before) &&
      getLastStoresInOfflArray(Dst);
}

bool OpenMPOpt::getLastAccessesToOfflArray(
    AllocaInst *OfflArray, SmallVectorImpl<Value *> &LastAccesses,
    Instruction *Before) {
  assert(OfflArray && "Can't get accesses to nullptr!");

  auto *DT =
      OMPInfoCache.getAnalysisResultForFunction<DominatorTreeAnalysis>(
          *OfflArray->getFunction());
  if (Before && !DT) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't find DominatorTreeAnalysis.\n");
    return false;
  }

  OfflArray->reverseUseList(); // To traverse users from top to bottom.
  for (auto *Usr : OfflArray->users()) {
    if (!isa<Instruction>(Usr))
      continue;

    auto *UsrInst = cast<Instruction>(Usr);

    // If reached lower limit.
    if (Before && !DT->dominates(UsrInst, Before))
      break;

    if (auto *Access = dyn_cast<GetElementPtrInst>(UsrInst)) {
      auto *ArrayIdx = Access->idx_begin() + 1;
      if (ArrayIdx == Access->idx_end())
        continue;

      const uint64_t IdxLiteral = getIntLiteral(ArrayIdx->get());
      LastAccesses[IdxLiteral] = UsrInst;
    } else if (UsrInst->isCast()){
      // If the access is directly a cast, it means it didn't need the
      // GEP to the array, which means, an access to the first position
      // of the array.
      LastAccesses[0] = UsrInst;
    } else {
      LLVM_DEBUG(dbgs() << TAG << "Unrecognized access pattern.\n");
      return false;
    }
  }

  if (!OffloadArray::isFilled(LastAccesses)) {
    LLVM_DEBUG(dbgs() << TAG << "Couldn't get last accesses to offload array.\n");
    return false;
  }
  return true;
}

bool OpenMPOpt::getLastStoresInOfflArray(MemoryTransfer::OffloadArray &Dst) {
  assert(Dst.isFilledLastAccesses() && "LastAccesses must be filled!");

  unsigned NumValues = Dst.size();
  auto &LastAccesses = Dst.LastAccesses;
  auto &StoredAddresses = Dst.StoredAddresses;

  for (unsigned It = 0; It < NumValues; ++It) {
    auto *Accs = LastAccesses[It];
    Accs->reverseUseList();
    auto AccsUsr = Accs->user_begin();
    if (AccsUsr == Accs->user_end()) {
      LLVM_DEBUG(dbgs() << TAG << "Useless access to offload array.\n");
      return false;
    }

    auto *I = cast<Instruction>(*AccsUsr);
    if (I->isCast())
      AccsUsr = I->user_begin();

    if (!isa<StoreInst>(*AccsUsr)) {
      LLVM_DEBUG(dbgs() << TAG << "Unrecognized access pattern.\n");
      return false;
    }

    StoredAddresses[It] =
        GetUnderlyingObject(AccsUsr->getOperand(0), OMPInfoCache.getDL());
  }

  if (!Dst.isFilledStoredAddresses()) {
    LLVM_DEBUG(dbgs() << TAG << "Didn't get last stores to offload array.\n");
    return false;
  }
  return true;
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
