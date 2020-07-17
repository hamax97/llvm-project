//===- OpenMPOptTest.h - Base file for OpenMPOpt unittests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TRANSFORMS_IPO_OPENMPOPT_H
#define LLVM_UNITTESTS_TRANSFORMS_IPO_OPENMPOPT_H

#include "llvm/Transforms/IPO/OpenMPOpt.h"
#include "llvm/AsmParser/Parser.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace llvm;
using namespace omp;

namespace {

using testing::NiceMock;
using testing::Matcher;
using testing::Return;
using testing::Invoke;
using testing::_;

template <typename DerivedT, typename IRUnitT,
    typename AnalysisManagerT = AnalysisManager<IRUnitT>,
    typename... ExtraArgTs>
class MockPassBase {
public:
  class Pass : public PassInfoMixin<Pass> {
    friend MockPassBase;

    DerivedT *Handle;

    Pass(DerivedT &Handle) : Handle(&Handle) {
      static_assert(std::is_base_of<MockPassBase, DerivedT>::value,
                    "Must pass the derived type to this template!");
    }

  public:
    PreservedAnalyses run(IRUnitT &IR, AnalysisManagerT &AM,
                          ExtraArgTs... ExtraArgs) {
      return Handle->run(IR, AM, ExtraArgs...);
    }
  };

  Pass getPass() { return Pass(static_cast<DerivedT &>(*this)); }

protected:
  /// Derived classes should call this in their constructor to set up default
  /// mock actions. (We can't do this in our constructor because this has to
  /// run after the DerivedT is constructed.)
  void setDefaults() {
    ON_CALL(static_cast<DerivedT &>(*this),
            run(_, _, testing::Matcher<ExtraArgTs>(_)...))
        .WillByDefault(Return(PreservedAnalyses::all()));
  }
};

struct MockFunctionPass
    : MockPassBase<MockFunctionPass, Function> {
  MOCK_METHOD2(run, PreservedAnalyses(Function &, FunctionAnalysisManager &));

  MockFunctionPass() { setDefaults(); }
};

struct MockModulePass : MockPassBase<MockModulePass, Module> {
  MOCK_METHOD2(run, PreservedAnalyses(Module &, ModuleAnalysisManager &));

  MockModulePass() { setDefaults(); }
};


struct MockSCCPass : MockPassBase<MockSCCPass, LazyCallGraph::SCC,
    CGSCCAnalysisManager &, LazyCallGraph &, CGSCCUpdateResult &> {
  MOCK_METHOD4(run,
               PreservedAnalyses(LazyCallGraph::SCC &, CGSCCAnalysisManager &,
                                 LazyCallGraph &, CGSCCUpdateResult &));

};

class OpenMPOptTest : public ::testing::Test {
protected:
  std::unique_ptr<LLVMContext> Ctx;

  OpenMPOptTest() : Ctx(new LLVMContext) {}

  std::unique_ptr<Module> parseModuleString(const char *ModuleString) {
    SMDiagnostic Err;
    auto M = parseAssemblyString(ModuleString, Err, *Ctx);
    EXPECT_TRUE(M);
    return M;
  }
};

} // end anonymous namespace

#endif // LLVM_UNITTESTS_TRANSFORMS_IPO_OPENMPOPT_H