//===- IPO/OpenMPOpt.h - Collection of OpenMP optimizations -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_OPENMP_OPT_H
#define LLVM_TRANSFORMS_IPO_OPENMP_OPT_H

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"
#include "llvm/Transforms/IPO/Attributor.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Analysis/MemorySSA.h"

namespace llvm {
namespace omp {

using namespace types;

/// OpenMP specific information. For now, stores RFIs and ICVs also needed for
/// Attributor runs.
struct OMPInformationCache : public InformationCache {
  OMPInformationCache(Module &M, AnalysisGetter &AG,
                      BumpPtrAllocator &Allocator, SetVector<Function *> *CGSCC,
                      SmallPtrSetImpl<Function *> &ModuleSlice)
      : InformationCache(M, AG, Allocator, CGSCC), ModuleSlice(ModuleSlice),
        OMPBuilder(M) {
    OMPBuilder.initialize();
    initializeRuntimeFunctions();
    initializeInternalControlVars();
  }

  /// Generic information that describes an internal control variable.
  struct InternalControlVarInfo {
    /// The kind, as described by InternalControlVar enum.
    InternalControlVar Kind;

    /// The name of the ICV.
    StringRef Name;

    /// Environment variable associated with this ICV.
    StringRef EnvVarName;

    /// Initial value kind.
    ICVInitValue InitKind;

    /// Initial value.
    ConstantInt *InitValue;

    /// Setter RTL function associated with this ICV.
    RuntimeFunction Setter;

    /// Getter RTL function associated with this ICV.
    RuntimeFunction Getter;

    /// RTL Function corresponding to the override clause of this ICV
    RuntimeFunction Clause;
  };

  /// Generic information that describes a runtime function
  struct RuntimeFunctionInfo {

    /// The kind, as described by the RuntimeFunction enum.
    RuntimeFunction Kind;

    /// The name of the function.
    StringRef Name;

    /// Flag to indicate a variadic function.
    bool IsVarArg;

    /// The return type of the function.
    Type *ReturnType;

    /// The argument types of the function.
    SmallVector<Type *, 8> ArgumentTypes;

    /// The declaration if available.
    Function *Declaration = nullptr;

    /// Uses of this runtime function per function containing the use.
    using UseVector = SmallVector<Use *, 16>;

    /// Return the vector of uses in function \p F.
    UseVector &getOrCreateUseVector(Function *F) {
      std::shared_ptr<UseVector> &UV = UsesMap[F];
      if (!UV)
        UV = std::make_shared<UseVector>();
      return *UV;
    }

    /// Return the vector of uses in function \p F or `nullptr` if there are
    /// none.
    const UseVector *getUseVector(Function &F) const {
      auto I = UsesMap.find(&F);
      if (I != UsesMap.end())
        return I->second.get();
      return nullptr;
    }

    /// Return how many functions contain uses of this runtime function.
    size_t getNumFunctionsWithUses() const { return UsesMap.size(); }

    /// Return the number of arguments (or the minimal number for variadic
    /// functions).
    size_t getNumArgs() const { return ArgumentTypes.size(); }

    /// Run the callback \p CB on each use and forget the use if the result is
    /// true. The callback will be fed the function in which the use was
    /// encountered as second argument.
    void foreachUse(function_ref<bool(Use &, Function &)> CB) {
      for (auto &It : UsesMap)
        foreachUse(CB, It.first, It.second.get());
    }

    /// Run the callback \p CB on each use within the function \p F and forget
    /// the use if the result is true.
    void foreachUse(function_ref<bool(Use &, Function &)> CB, Function *F,
                    UseVector *Uses = nullptr);

  private:
    /// Map from functions to all uses of this runtime function contained in
    /// them.
    DenseMap<Function *, std::shared_ptr<UseVector>> UsesMap;
  };

  /// Used to store information about a runtime call that involves
  /// host to device memory offloading.
  struct MemoryTransfer {
    struct OffloadArray {
      AllocaInst *Array;
      SmallVector<Value *, 8> LastAccesses;
      SmallVector<Value *, 8> StoredValues;
      InformationCache &InfoCache;

      OffloadArray(AllocaInst *Array, InformationCache &InfoCache)
          : Array(Array), InfoCache(InfoCache) {}

      bool getValues(Instruction *Before = nullptr);

    private:
      bool getLastAccessesToOfflArray(Instruction *Before = nullptr);

      bool getLastStoresInOfflArray();

      void initialize(unsigned Size, Value *Values = nullptr) {
        LastAccesses.assign(Size, Values);
        StoredValues.assign(Size, Values);
      }
      unsigned numValues() { return LastAccesses.size(); }
      static bool isFilled(const SmallVectorImpl<Value *> &V);
    };

    CallBase *RuntimeCall;
    InformationCache &InfoCache;
    std::unique_ptr<OffloadArray> BasePtrs;
    std::unique_ptr<OffloadArray> Ptrs;
    std::unique_ptr<OffloadArray> Sizes;

    MemoryTransfer(CallBase *RuntimeCall, InformationCache &InfoCache) :
        RuntimeCall{RuntimeCall}, InfoCache{InfoCache}
    {}

    /// Gets the values stored in the offload arrays. Returns false if some of
    /// the values couldn't be found.
    bool getValuesInOfflArrays();
  };

  /// The slice of the module we are allowed to look at.
  SmallPtrSetImpl<Function *> &ModuleSlice;

  /// An OpenMP-IR-Builder instance
  OpenMPIRBuilder OMPBuilder;

  /// Map from runtime function kind to the runtime function description.
  EnumeratedArray<RuntimeFunctionInfo, RuntimeFunction,
      RuntimeFunction::OMPRTL___last>
      RFIs;

  /// Map from ICV kind to the ICV description.
  EnumeratedArray<InternalControlVarInfo, InternalControlVar,
      InternalControlVar::ICV___last>
      ICVs;

  /// Helper to initialize all internal control variable information for those
  /// defined in OMPKinds.def.
  void initializeInternalControlVars();

  /// Helper to initialize all runtime function information for those defined
  /// in OpenMPKinds.def.
  void initializeRuntimeFunctions();

  /// Returns true if the function declaration \p F matches the runtime
  /// function types, that is, return type \p RTFRetType, and argument types
  /// \p RTFArgTypes.
  static bool declMatchesRTFTypes(Function *F, Type *RTFRetType,
                                  SmallVector<Type *, 8> &RTFArgTypes);
};

struct OpenMPOpt {

  using MemoryTransfer = OMPInformationCache::MemoryTransfer;
  using OptimizationRemarkGetter =
  function_ref<OptimizationRemarkEmitter &(Function *)>;

  OpenMPOpt(SmallVectorImpl<Function *> &SCC, CallGraphUpdater &CGUpdater,
            OptimizationRemarkGetter OREGetter,
            OMPInformationCache &OMPInfoCache, Attributor &A)
      : M(*(*SCC.begin())->getParent()), SCC(SCC), CGUpdater(CGUpdater),
        OREGetter(OREGetter), OMPInfoCache(OMPInfoCache), A(A) {}

  /// Run all OpenMP optimizations on the underlying SCC/ModuleSlice.
  bool run();

  /// Return the call if \p U is a callee use in a regular call. If \p RFI is
  /// given it has to be the callee or a nullptr is returned.
  static CallInst *getCallIfRegularCall(
      Use &U, OMPInformationCache::RuntimeFunctionInfo *RFI = nullptr);

  /// Return the call if \p V is a regular call. If \p RFI is given it has to be
  /// the callee or a nullptr is returned.
  static CallInst *getCallIfRegularCall(
      Value &V, OMPInformationCache::RuntimeFunctionInfo *RFI = nullptr);

  /// Returns the integer representation of \p V.
  static uint64_t getIntLiteral(const Value *V) {
    assert(V && "Getting Integer value of nullptr");
    return (dyn_cast<ConstantInt>(V))->getZExtValue();
  }

private:
  /// Try to delete parallel regions if possible.
  bool deleteParallelRegions();

  /// Try to eliminiate runtime calls by reusing existing ones.
  bool deduplicateRuntimeCalls();

  /// Splits a runtime call that involves a host to device transfer into its ""
  bool splitMemoryTransfer(MemoryTransfer &MT);

  static Value *combinedIdentStruct(Value *CurrentIdent, Value *NextIdent,
                                    bool GlobalOnly, bool &SingleChoice);

  /// Return an `struct ident_t*` value that represents the ones used in the
  /// calls of \p RFI inside of \p F. If \p GlobalOnly is true, we will not
  /// return a local `struct ident_t*`. For now, if we cannot find a suitable
  /// return value we create one from scratch. We also do not yet combine
  /// information, e.g., the source locations, see combinedIdentStruct.
  Value *
  getCombinedIdentFromCallUsesIn(OMPInformationCache::RuntimeFunctionInfo &RFI,
                                 Function &F, bool GlobalOnly);

  /// Try to eliminiate calls of \p RFI in \p F by reusing an existing one or
  /// \p ReplVal if given.
  bool deduplicateRuntimeCalls(Function &F,
                               OMPInformationCache::RuntimeFunctionInfo &RFI,
                               Value *ReplVal = nullptr);

  /// Tries to hide the latency of runtime calls that involve host to
  /// device memory transfers.
  bool hideMemTransfersLatency();

  /// Collect arguments that represent the global thread id in \p GTIdArgs.
  void collectGlobalThreadIdArguments(SmallSetVector<Value *, 16> &GTIdArgs);

  /// Emit a remark generically
  ///
  /// This template function can be used to generically emit a remark. The
  /// RemarkKind should be one of the following:
  ///   - OptimizationRemark to indicate a successful optimization attempt
  ///   - OptimizationRemarkMissed to report a failed optimization attempt
  ///   - OptimizationRemarkAnalysis to provide additional information about an
  ///     optimization attempt
  ///
  /// The remark is built using a callback function provided by the caller that
  /// takes a RemarkKind as input and returns a RemarkKind.
  template <typename RemarkKind,
      typename RemarkCallBack = function_ref<RemarkKind(RemarkKind &&)>>
  void emitRemark(Instruction *Inst, StringRef RemarkName,
                  RemarkCallBack &&RemarkCB);

  /// Emit a remark on a function. Since only OptimizationRemark is supporting
  /// this, it can't be made generic.
  void emitRemarkOnFunction(
      Function *F, StringRef RemarkName,
      function_ref<OptimizationRemark(OptimizationRemark &&)> &&RemarkCB);

  /// The underyling module.
  Module &M;

  /// The SCC we are operating on.
  SmallVectorImpl<Function *> &SCC;

  /// Callback to update the call graph, the first argument is a removed call,
  /// the second an optional replacement call.
  CallGraphUpdater &CGUpdater;

  /// Callback to get an OptimizationRemarkEmitter from a Function *
  OptimizationRemarkGetter OREGetter;

  /// OpenMP-specific information cache. Also Used for Attributor runs.
  OMPInformationCache &OMPInfoCache;

  /// Attributor instance.
  Attributor &A;

  /// Helper function to run Attributor on SCC.
  bool runAttributor();

  /// Populate the Attributor with abstract attribute opportunities in the
  /// function.
  void registerAAs();
};

/// Helper to remember if the module contains OpenMP (runtime calls), to be used
/// foremost with containsOpenMP.
struct OpenMPInModule {
  OpenMPInModule &operator=(bool Found) {
    if (Found)
      Value = OpenMPInModule::OpenMP::FOUND;
    else
      Value = OpenMPInModule::OpenMP::NOT_FOUND;
    return *this;
  }
  bool isKnown() { return Value != OpenMP::UNKNOWN; }
  operator bool() { return Value != OpenMP::NOT_FOUND; }

private:
  enum class OpenMP { FOUND, NOT_FOUND, UNKNOWN } Value = OpenMP::UNKNOWN;
};

/// Helper to determine if \p M contains OpenMP (runtime calls).
bool containsOpenMP(Module &M, OpenMPInModule &OMPInModule);

} // namespace omp

/// OpenMP optimizations pass.
class OpenMPOptPass : public PassInfoMixin<OpenMPOptPass> {
  /// Helper to remember if the module contains OpenMP (runtime calls).
  omp::OpenMPInModule OMPInModule;

public:
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                        LazyCallGraph &CG, CGSCCUpdateResult &UR);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_OPENMP_OPT_H
