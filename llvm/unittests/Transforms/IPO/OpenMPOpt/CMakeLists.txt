set(LLVM_LINK_COMPONENTS
  ipo
  Core
  Analysis
  AsmParser
  FrontendOpenMP
  TransformUtils
  )

add_llvm_unittest(OpenMPOptUnitTests
  HideMemTransferLatencyTest.cpp
  )