# Package

version       = "0.1.0"
author        = "Dmitry Dorofeev"
description   = "Nim lang version of the code for the Neural Networks and Deep Learning free book "
license       = "MIT"
srcDir        = "src"
bin           = @["deep_book"]



# Dependencies

requires "nim >= 0.20.0",
  "arraymancer 0.5.2",
  "zero_functional 0.3.0"

