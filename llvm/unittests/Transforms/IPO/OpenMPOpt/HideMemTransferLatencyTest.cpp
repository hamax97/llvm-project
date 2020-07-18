//===- HideMemTransferLatencyTest.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OpenMPOptTest.h"

namespace {

class HideMemTransferLatencyTest : public OpenMPOptTest {
protected:
  NiceMock<MockFunctionPass> MFP;
  NiceMock<MockModulePass> MMP;
  MockSCCPass MSCCP;
  ModulePassManager MPM;

  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;
  CGSCCAnalysisManager CGAM;

  HideMemTransferLatencyTest()
      : OpenMPOptTest(), FAM(true), MAM(true), CGAM(true) {
    MAM.registerPass([] { return LazyCallGraphAnalysis(); });
    FAM.registerPass([] { return TargetLibraryAnalysis(); });
    FAM.registerPass([]{ return AAManager(); });
    FAM.registerPass([]{ return DominatorTreeAnalysis(); });
    FAM.registerPass([]{ return MemorySSAAnalysis(); });
    CGAM.registerPass([&] { return FunctionAnalysisManagerCGSCCProxy(); });

    // Register required pass instrumentation analysis.
    FAM.registerPass([] { return PassInstrumentationAnalysis(); });
    MAM.registerPass([] { return PassInstrumentationAnalysis(); });
    CGAM.registerPass([] {return PassInstrumentationAnalysis(); });

    // Cross-register proxies.
    MAM.registerPass([&] { return FunctionAnalysisManagerModuleProxy(FAM); });
    MAM.registerPass([&] { return CGSCCAnalysisManagerModuleProxy(CGAM); });
    CGAM.registerPass([&] { return ModuleAnalysisManagerCGSCCProxy(MAM); });
    FAM.registerPass([&] { return CGSCCAnalysisManagerFunctionProxy(CGAM); });
    FAM.registerPass([&] { return ModuleAnalysisManagerFunctionProxy(MAM); });

//    FunctionPassManager FPM;
//    FPM.addPass(MFP.getPass());

    CGSCCPassManager CGPM;
    CGPM.addPass(MSCCP.getPass());
//    CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(FPM)));

    //MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
  }
};

TEST_F(HideMemTransferLatencyTest, GetValuesInOfflArrays) {
  const char *ModuleString =
      "@.offload_maptypes = private unnamed_addr constant [1 x i64] [i64 33]\n"
      "define dso_local i32 @dataTransferOnly(double* noalias %a, i32 %size) {\n"
      "entry:\n"
      "  %.offload_baseptrs = alloca [1 x i8*], align 8\n"
      "  %.offload_ptrs = alloca [1 x i8*], align 8\n"
      "  %.offload_sizes = alloca [1 x i64], align 8\n"

      "  %call = call i32 @rand()\n"

      "  %conv = zext i32 %size to i64\n"
      "  %0 = shl nuw nsw i64 %conv, 3\n"
      "  %1 = getelementptr inbounds [1 x i8*], [1 x i8*]* %.offload_baseptrs, i64 0, i64 0\n"
      "  %2 = bitcast [1 x i8*]* %.offload_baseptrs to double**\n"
      "  store double* %a, double** %2, align 8\n"
      "  %3 = getelementptr inbounds [1 x i8*], [1 x i8*]* %.offload_ptrs, i64 0, i64 0\n"
      "  %4 = bitcast [1 x i8*]* %.offload_ptrs to double**\n"
      "  store double* %a, double** %4, align 8\n"
      "  %5 = getelementptr inbounds [1 x i64], [1 x i64]* %.offload_sizes, i64 0, i64 0\n"
      "  store i64 %0, i64* %5, align 8\n"
      "  call void @__tgt_target_data_begin(i64 -1, i32 1, i8** nonnull %1, i8** nonnull %3, i64* nonnull %5, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @.offload_maptypes, i64 0, i64 0))\n"

      "  %rem = urem i32 %call, %size\n"

      "  call void @__tgt_target_data_end(i64 -1, i32 1, i8** nonnull %1, i8** nonnull %3, i64* nonnull %5, i64* getelementptr inbounds ([1 x i64], [1 x i64]* @.offload_maptypes, i64 0, i64 0))\n"
      "  ret i32 %rem\n"
      "}\n"

      "declare dso_local void @__tgt_target_data_begin(i64, i32, i8**, i8**, i64*, i64*)\n"
      "declare dso_local void @__tgt_target_data_end(i64, i32, i8**, i8**, i64*, i64*)\n"
      "declare dso_local i32 @rand()";

  std::unique_ptr<Module> M = parseModuleString(ModuleString);

  EXPECT_CALL(MSCCP, run(_, _, _, _))
      .WillRepeatedly(Invoke([](LazyCallGraph::SCC &C, CGSCCAnalysisManager &AM,
                                LazyCallGraph &CG, CGSCCUpdateResult &UR) {
        SmallPtrSet<Function *, 16> ModuleSlice;
        SmallVector<Function *, 16> SCC;
        for (LazyCallGraph::Node &N : C) {
          SCC.push_back(&N.getFunction());
          ModuleSlice.insert(SCC.back());
        }

        EXPECT_FALSE(SCC.empty());

        FunctionAnalysisManager &FAM =
            AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

        AnalysisGetter AG(FAM);
        CallGraphUpdater CGUpdater;
        CGUpdater.initialize(CG, C, AM, UR);

        SetVector<Function *> Functions(SCC.begin(), SCC.end());
        BumpPtrAllocator Allocator;
        OMPInformationCache InfoCache(*(Functions.back()->getParent()), AG, Allocator,
            /*CGSCC*/ &Functions, ModuleSlice);
        Attributor A(Functions, InfoCache, CGUpdater);

        auto &RFI = InfoCache.RFIs[OMPRTL___tgt_target_data_begin];
        auto GetValuesInOfflArrays = [&](Use &U, Function &Decl) {
          auto *RTCall = OpenMPOpt::getCallIfRegularCall(U, &RFI);
          EXPECT_TRUE(RTCall);

          OMPInformationCache::MemoryTransfer MT(RTCall, InfoCache);
          bool Success = MT.getValuesInOfflArrays();
          EXPECT_TRUE(Success);

          std::string ValueName;
          raw_string_ostream OS(ValueName);
          // Check for **offload_baseptrs.
          auto &BasePtrsValues = MT.BasePtrs->StoredValues;
          EXPECT_EQ(BasePtrsValues.size(), (size_t) 1);
          BasePtrsValues[0]->print(OS);
          EXPECT_STREQ(OS.str().c_str(), "double* %a");
          ValueName.clear();

          // Check for **offload_ptrs.
          auto &PtrsValues = MT.Ptrs->StoredValues;
          EXPECT_EQ(PtrsValues.size(), (size_t) 1);
          PtrsValues[0]->print(OS);
          EXPECT_STREQ(OS.str().c_str(), "double* %a");
          ValueName.clear();

          // Check for **offload_sizes.
          auto &SizesValues = MT.Sizes->StoredValues;
          EXPECT_EQ(SizesValues.size(), (size_t) 1);
          SizesValues[0]->print(OS);
          EXPECT_STREQ(OS.str().c_str(), "  %0 = shl nuw nsw i64 %conv, 3");

          return false;
        };
        RFI.foreachUse(GetValuesInOfflArrays);

        return PreservedAnalyses::all();
      }));

  MPM.run(*M, MAM);
}

} // end anonymous namespace