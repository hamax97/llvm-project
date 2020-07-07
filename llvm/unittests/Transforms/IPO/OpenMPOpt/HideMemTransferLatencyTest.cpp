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
  /// This stores the runtime calls for further use within each test.
  SmallVector<CallBase *, 16> MemTransferCalls;

  void getCallSites(const char *ModuleString) {
    auto M = parseModuleString(ModuleString);
    initializeOMPInfoCache(*M);

    auto &RFI = OMPInfoCache->RFIs[OMPRTL___tgt_target_data_begin];
    auto GetCallSites = [&](Use &U, Function &Decl) {
      auto *RTCall = OMPInformationCache::getCallIfRegularCall(U, &RFI);
      if (!RTCall)
        return false;
      MemTransferCalls.push_back(RTCall);
      return false;
    };
    RFI.foreachUse(GetCallSites);
    EXPECT_FALSE(MemTransferCalls.empty());
  }
};

TEST_F(HideMemTransferLatencyTest, GetValuesInOfflArrays) {
// TODO: Update this ModuleString.
const char *ModuleString = "";
getCallSites(ModuleString);

EXPECT_TRUE(true);
}

} // end anonymous namespace