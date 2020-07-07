//===- OpenMPOptTest.h - Base file for OpenMPOpt unittests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TRANSFORMS_IPO_OPENMPOPT_H
#define LLVM_UNITTESTS_TRANSFORMS_IPO_OPENMPOPT_H

#include "../../../../lib/Transforms/IPO/OpenMPOptPriv.h"
#include "llvm/AsmParser/Parser.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class OpenMPOptTest : public ::testing::Test {
protected:
  std::unique_ptr<OMPInformationCache> OMPInfoCache;
  std::unique_ptr<LLVMContext> Ctx;

  OpenMPOptTest() : Ctx(new LLVMContext) {}

  /// Initializes OMPInfoCache attribute cache given a module \p M.
  /// This must be called before using OMPInfoCache.
  void initializeOMPInfoCache(Module &M) {
    CallGraph CG(M);
    scc_iterator<CallGraph *> CGI = scc_begin(&CG);
    CallGraphSCC CGSCC(CG, &CGI);

    SmallPtrSet<Function *, 16> ModuleSlice;
    SmallVector<Function *, 16> SCC;
    for (CallGraphNode *CGN : CGSCC)
      if (Function *Fn = CGN->getFunction())
        if (!Fn->isDeclaration()) {
          SCC.push_back(Fn);
          ModuleSlice.insert(Fn);
        }

    EXPECT_FALSE(SCC.empty());

    AnalysisGetter AG;
    SetVector<Function *> Functions(SCC.begin(), SCC.end());
    BumpPtrAllocator Allocator;
    OMPInfoCache = std::make_unique<OMPInformationCache>(
        *(Functions.back()->getParent()), AG, Allocator, /*CGSCC*/ &Functions,
        ModuleSlice);
  }

  std::unique_ptr<Module> parseModuleString(const char *ModuleString) {
    SMDiagnostic Err;
    auto M = parseAssemblyString(ModuleString, Err, *Ctx);
    EXPECT_TRUE(M);
    return M;
  }
};

} // end anonymous namespace

#endif // LLVM_UNITTESTS_TRANSFORMS_IPO_OPENMPOPT_H