import argparse
import os
import re

from alt_wkl_configs import *
import ast
from get_output_sizes import calc_conv_output_size, calc_maxpool_output_size
from vta import environment
env = environment.get_env()

mod_string = """#[version = "0.0.5"]
def @main(%input0: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {
  %0 = nn.conv2d(%input0, meta[relay.Constant][0] /* ty=Tensor[(32, 3, 3, 3), float32] */, strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %1 = add(%0, meta[relay.Constant][1] /* ty=Tensor[(32, 1, 1), float32] */) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %2 = clip(%1, a_min=0f, a_max=6f) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %3 = annotation.stop_fusion(%2) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %4 = multiply(%3, 16f /* ty=float32 */) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %5 = round(%4) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %6 = clip(%5, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 32, 112, 112), float32] */;
  %7 = cast(%6, dtype="int8") /* ty=Tensor[(1, 32, 112, 112), int8] */;
  %8 = nn.conv2d(%7, meta[relay.Constant][2] /* ty=Tensor[(32, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=32, channels=32, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 32, 112, 112), int32] */;
  %9 = add(%8, meta[relay.Constant][3] /* ty=Tensor[(32, 1, 1), int32] */) /* ty=Tensor[(1, 32, 112, 112), int32] */;
  %10 = clip(%9, a_min=0f, a_max=24576f) /* ty=Tensor[(1, 32, 112, 112), int32] */;
  %11 = add(%10, 128 /* ty=int32 */) /* ty=Tensor[(1, 32, 112, 112), int32] */;
  %12 = right_shift(%11, 8 /* ty=int32 */) /* ty=Tensor[(1, 32, 112, 112), int32] */;
  %13 = clip(%12, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 32, 112, 112), int32] */;
  %14 = cast(%13, dtype="int8") /* ty=Tensor[(1, 32, 112, 112), int8] */;
  %15 = annotation.stop_fusion(%14) /* ty=Tensor[(1, 32, 112, 112), int8] */;
  %16 = nn.conv2d(%15, meta[relay.Constant][4] /* ty=Tensor[(16, 32, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 16, 112, 112), int32] */;
  %17 = add(%16, meta[relay.Constant][5] /* ty=Tensor[(16, 1, 1), int32] */) /* ty=Tensor[(1, 16, 112, 112), int32] */;
  %18 = add(%17, 32 /* ty=int32 */) /* ty=Tensor[(1, 16, 112, 112), int32] */;
  %19 = right_shift(%18, 6 /* ty=int32 */) /* ty=Tensor[(1, 16, 112, 112), int32] */;
  %20 = clip(%19, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 16, 112, 112), int32] */;
  %21 = cast(%20, dtype="int8") /* ty=Tensor[(1, 16, 112, 112), int8] */;
  %22 = annotation.stop_fusion(%21) /* ty=Tensor[(1, 16, 112, 112), int8] */;
  %23 = nn.conv2d(%22, meta[relay.Constant][6] /* ty=Tensor[(96, 16, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 96, 112, 112), int32] */;
  %24 = left_shift(%23, 21 /* ty=int32 */) /* ty=Tensor[(1, 96, 112, 112), int32] */;
  %25 = add(%24, meta[relay.Constant][7] /* ty=Tensor[(96, 1, 1), int32] */) /* ty=Tensor[(1, 96, 112, 112), int32] */;
  %26 = clip(%25, a_min=0f, a_max=2.57698e+10f) /* ty=Tensor[(1, 96, 112, 112), int32] */;
  %27 = add(%26, 134217728 /* ty=int32 */) /* ty=Tensor[(1, 96, 112, 112), int32] */;
  %28 = right_shift(%27, 28 /* ty=int32 */) /* ty=Tensor[(1, 96, 112, 112), int32] */;
  %29 = clip(%28, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 96, 112, 112), int32] */;
  %30 = cast(%29, dtype="int8") /* ty=Tensor[(1, 96, 112, 112), int8] */;
  %31 = annotation.stop_fusion(%30) /* ty=Tensor[(1, 96, 112, 112), int8] */;
  %32 = nn.conv2d(%31, meta[relay.Constant][8] /* ty=Tensor[(96, 1, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], groups=96, channels=96, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 96, 56, 56), int32] */;
  %33 = add(%32, meta[relay.Constant][9] /* ty=Tensor[(96, 1, 1), int32] */) /* ty=Tensor[(1, 96, 56, 56), int32] */;
  %34 = clip(%33, a_min=0f, a_max=49152f) /* ty=Tensor[(1, 96, 56, 56), int32] */;
  %35 = add(%34, 256 /* ty=int32 */) /* ty=Tensor[(1, 96, 56, 56), int32] */;
  %36 = right_shift(%35, 9 /* ty=int32 */) /* ty=Tensor[(1, 96, 56, 56), int32] */;
  %37 = clip(%36, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 96, 56, 56), int32] */;
  %38 = cast(%37, dtype="int8") /* ty=Tensor[(1, 96, 56, 56), int8] */;
  %39 = annotation.stop_fusion(%38) /* ty=Tensor[(1, 96, 56, 56), int8] */;
  %40 = nn.conv2d(%39, meta[relay.Constant][10] /* ty=Tensor[(24, 96, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %41 = add(%40, meta[relay.Constant][11] /* ty=Tensor[(24, 1, 1), int32] */) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %42 = add(%41, 64 /* ty=int32 */) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %43 = right_shift(%42, 7 /* ty=int32 */) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %44 = clip(%43, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %45 = cast(%44, dtype="int8") /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %46 = annotation.stop_fusion(%45) /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %47 = cast(%46, dtype="int32") /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %48 = cast(%44, dtype="int8") /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %49 = annotation.stop_fusion(%48) /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %50 = nn.conv2d(%49, meta[relay.Constant][12] /* ty=Tensor[(144, 24, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %51 = left_shift(%50, 20 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %52 = add(%51, meta[relay.Constant][13] /* ty=Tensor[(144, 1, 1), int32] */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %53 = clip(%52, a_min=0f, a_max=2.57698e+10f) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %54 = add(%53, 134217728 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %55 = right_shift(%54, 28 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %56 = clip(%55, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %57 = cast(%56, dtype="int8") /* ty=Tensor[(1, 144, 56, 56), int8] */;
  %58 = annotation.stop_fusion(%57) /* ty=Tensor[(1, 144, 56, 56), int8] */;
  %59 = nn.conv2d(%58, meta[relay.Constant][14] /* ty=Tensor[(144, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=144, channels=144, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %60 = add(%59, meta[relay.Constant][15] /* ty=Tensor[(144, 1, 1), int32] */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %61 = clip(%60, a_min=0f, a_max=49152f) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %62 = add(%61, 256 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %63 = right_shift(%62, 9 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %64 = clip(%63, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %65 = cast(%64, dtype="int8") /* ty=Tensor[(1, 144, 56, 56), int8] */;
  %66 = annotation.stop_fusion(%65) /* ty=Tensor[(1, 144, 56, 56), int8] */;
  %67 = nn.conv2d(%66, meta[relay.Constant][16] /* ty=Tensor[(24, 144, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %68 = add(%67, meta[relay.Constant][17] /* ty=Tensor[(24, 1, 1), int32] */) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %69 = add(%68, 64 /* ty=int32 */) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %70 = right_shift(%69, 7 /* ty=int32 */) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %71 = clip(%70, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %72 = cast(%71, dtype="int8") /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %73 = annotation.stop_fusion(%72) /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %74 = cast(%73, dtype="int32") /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %75 = add(%47, %74) /* ty=Tensor[(1, 24, 56, 56), int32] */;
  %76 = cast(%75, dtype="int8") /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %77 = annotation.stop_fusion(%76) /* ty=Tensor[(1, 24, 56, 56), int8] */;
  %78 = nn.conv2d(%77, meta[relay.Constant][18] /* ty=Tensor[(144, 24, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %79 = left_shift(%78, 19 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %80 = add(%79, meta[relay.Constant][19] /* ty=Tensor[(144, 1, 1), int32] */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %81 = clip(%80, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %82 = add(%81, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %83 = right_shift(%82, 27 /* ty=int32 */) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %84 = clip(%83, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 144, 56, 56), int32] */;
  %85 = cast(%84, dtype="int8") /* ty=Tensor[(1, 144, 56, 56), int8] */;
  %86 = annotation.stop_fusion(%85) /* ty=Tensor[(1, 144, 56, 56), int8] */;
  %87 = nn.conv2d(%86, meta[relay.Constant][20] /* ty=Tensor[(144, 1, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], groups=144, channels=144, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 144, 28, 28), int32] */;
  %88 = add(%87, meta[relay.Constant][21] /* ty=Tensor[(144, 1, 1), int32] */) /* ty=Tensor[(1, 144, 28, 28), int32] */;
  %89 = clip(%88, a_min=0f, a_max=49152f) /* ty=Tensor[(1, 144, 28, 28), int32] */;
  %90 = add(%89, 256 /* ty=int32 */) /* ty=Tensor[(1, 144, 28, 28), int32] */;
  %91 = right_shift(%90, 9 /* ty=int32 */) /* ty=Tensor[(1, 144, 28, 28), int32] */;
  %92 = clip(%91, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 144, 28, 28), int32] */;
  %93 = cast(%92, dtype="int8") /* ty=Tensor[(1, 144, 28, 28), int8] */;
  %94 = annotation.stop_fusion(%93) /* ty=Tensor[(1, 144, 28, 28), int8] */;
  %95 = nn.conv2d(%94, meta[relay.Constant][22] /* ty=Tensor[(32, 144, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %96 = add(%95, meta[relay.Constant][23] /* ty=Tensor[(32, 1, 1), int32] */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %97 = add(%96, 64 /* ty=int32 */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %98 = right_shift(%97, 7 /* ty=int32 */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %99 = clip(%98, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %100 = cast(%99, dtype="int8") /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %101 = annotation.stop_fusion(%100) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %102 = cast(%101, dtype="int32") /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %103 = cast(%99, dtype="int8") /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %104 = annotation.stop_fusion(%103) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %105 = nn.conv2d(%104, meta[relay.Constant][24] /* ty=Tensor[(192, 32, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %106 = left_shift(%105, 21 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %107 = add(%106, meta[relay.Constant][25] /* ty=Tensor[(192, 1, 1), int32] */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %108 = clip(%107, a_min=0f, a_max=5.15396e+10f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %109 = add(%108, 268435456 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %110 = right_shift(%109, 29 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %111 = clip(%110, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %112 = cast(%111, dtype="int8") /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %113 = annotation.stop_fusion(%112) /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %114 = nn.conv2d(%113, meta[relay.Constant][26] /* ty=Tensor[(192, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %115 = add(%114, meta[relay.Constant][27] /* ty=Tensor[(192, 1, 1), int32] */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %116 = clip(%115, a_min=0f, a_max=49152f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %117 = add(%116, 256 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %118 = right_shift(%117, 9 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %119 = clip(%118, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %120 = cast(%119, dtype="int8") /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %121 = annotation.stop_fusion(%120) /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %122 = nn.conv2d(%121, meta[relay.Constant][28] /* ty=Tensor[(32, 192, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %123 = add(%122, meta[relay.Constant][29] /* ty=Tensor[(32, 1, 1), int32] */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %124 = add(%123, 64 /* ty=int32 */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %125 = right_shift(%124, 7 /* ty=int32 */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %126 = clip(%125, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %127 = cast(%126, dtype="int8") /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %128 = annotation.stop_fusion(%127) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %129 = cast(%128, dtype="int32") /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %130 = add(%102, %129) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %131 = clip(%130, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %132 = cast(%131, dtype="int8") /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %133 = annotation.stop_fusion(%132) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %134 = cast(%133, dtype="int32") /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %135 = cast(%131, dtype="int8") /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %136 = annotation.stop_fusion(%135) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %137 = nn.conv2d(%136, meta[relay.Constant][30] /* ty=Tensor[(192, 32, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %138 = left_shift(%137, 20 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %139 = add(%138, meta[relay.Constant][31] /* ty=Tensor[(192, 1, 1), int32] */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %140 = clip(%139, a_min=0f, a_max=2.57698e+10f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %141 = add(%140, 134217728 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %142 = right_shift(%141, 28 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %143 = clip(%142, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %144 = cast(%143, dtype="int8") /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %145 = annotation.stop_fusion(%144) /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %146 = nn.conv2d(%145, meta[relay.Constant][32] /* ty=Tensor[(192, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %147 = add(%146, meta[relay.Constant][33] /* ty=Tensor[(192, 1, 1), int32] */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %148 = clip(%147, a_min=0f, a_max=49152f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %149 = add(%148, 256 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %150 = right_shift(%149, 9 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %151 = clip(%150, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %152 = cast(%151, dtype="int8") /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %153 = annotation.stop_fusion(%152) /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %154 = nn.conv2d(%153, meta[relay.Constant][34] /* ty=Tensor[(32, 192, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %155 = add(%154, meta[relay.Constant][35] /* ty=Tensor[(32, 1, 1), int32] */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %156 = add(%155, 64 /* ty=int32 */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %157 = right_shift(%156, 7 /* ty=int32 */) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %158 = clip(%157, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %159 = cast(%158, dtype="int8") /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %160 = annotation.stop_fusion(%159) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %161 = cast(%160, dtype="int32") /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %162 = add(%134, %161) /* ty=Tensor[(1, 32, 28, 28), int32] */;
  %163 = cast(%162, dtype="int8") /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %164 = annotation.stop_fusion(%163) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %165 = nn.conv2d(%164, meta[relay.Constant][36] /* ty=Tensor[(192, 32, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %166 = left_shift(%165, 20 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %167 = add(%166, meta[relay.Constant][37] /* ty=Tensor[(192, 1, 1), int32] */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %168 = clip(%167, a_min=0f, a_max=2.57698e+10f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %169 = add(%168, 134217728 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %170 = right_shift(%169, 28 /* ty=int32 */) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %171 = clip(%170, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 192, 28, 28), int32] */;
  %172 = cast(%171, dtype="int8") /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %173 = annotation.stop_fusion(%172) /* ty=Tensor[(1, 192, 28, 28), int8] */;
  %174 = nn.conv2d(%173, meta[relay.Constant][38] /* ty=Tensor[(192, 1, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 192, 14, 14), int32] */;
  %175 = add(%174, meta[relay.Constant][39] /* ty=Tensor[(192, 1, 1), int32] */) /* ty=Tensor[(1, 192, 14, 14), int32] */;
  %176 = clip(%175, a_min=0f, a_max=49152f) /* ty=Tensor[(1, 192, 14, 14), int32] */;
  %177 = add(%176, 256 /* ty=int32 */) /* ty=Tensor[(1, 192, 14, 14), int32] */;
  %178 = right_shift(%177, 9 /* ty=int32 */) /* ty=Tensor[(1, 192, 14, 14), int32] */;
  %179 = clip(%178, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 192, 14, 14), int32] */;
  %180 = cast(%179, dtype="int8") /* ty=Tensor[(1, 192, 14, 14), int8] */;
  %181 = annotation.stop_fusion(%180) /* ty=Tensor[(1, 192, 14, 14), int8] */;
  %182 = nn.conv2d(%181, meta[relay.Constant][40] /* ty=Tensor[(64, 192, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %183 = add(%182, meta[relay.Constant][41] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %184 = add(%183, 64 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %185 = right_shift(%184, 7 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %186 = clip(%185, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %187 = cast(%186, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %188 = annotation.stop_fusion(%187) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %189 = cast(%188, dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %190 = cast(%186, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %191 = annotation.stop_fusion(%190) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %192 = nn.conv2d(%191, meta[relay.Constant][42] /* ty=Tensor[(384, 64, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %193 = left_shift(%192, 20 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %194 = add(%193, meta[relay.Constant][43] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %195 = clip(%194, a_min=0f, a_max=2.57698e+10f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %196 = add(%195, 134217728 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %197 = right_shift(%196, 28 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %198 = clip(%197, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %199 = cast(%198, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %200 = annotation.stop_fusion(%199) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %201 = nn.conv2d(%200, meta[relay.Constant][44] /* ty=Tensor[(384, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %202 = add(%201, meta[relay.Constant][45] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %203 = clip(%202, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %204 = add(%203, 512 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %205 = right_shift(%204, 10 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %206 = clip(%205, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %207 = cast(%206, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %208 = annotation.stop_fusion(%207) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %209 = nn.conv2d(%208, meta[relay.Constant][46] /* ty=Tensor[(64, 384, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %210 = add(%209, meta[relay.Constant][47] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %211 = add(%210, 128 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %212 = right_shift(%211, 8 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %213 = clip(%212, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %214 = cast(%213, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %215 = annotation.stop_fusion(%214) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %216 = cast(%215, dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %217 = add(%189, %216) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %218 = clip(%217, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %219 = cast(%218, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %220 = annotation.stop_fusion(%219) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %221 = cast(%220, dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %222 = cast(%218, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %223 = annotation.stop_fusion(%222) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %224 = nn.conv2d(%223, meta[relay.Constant][48] /* ty=Tensor[(384, 64, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %225 = left_shift(%224, 20 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %226 = add(%225, meta[relay.Constant][49] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %227 = clip(%226, a_min=0f, a_max=2.57698e+10f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %228 = add(%227, 134217728 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %229 = right_shift(%228, 28 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %230 = clip(%229, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %231 = cast(%230, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %232 = annotation.stop_fusion(%231) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %233 = nn.conv2d(%232, meta[relay.Constant][50] /* ty=Tensor[(384, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %234 = add(%233, meta[relay.Constant][51] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %235 = clip(%234, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %236 = add(%235, 512 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %237 = right_shift(%236, 10 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %238 = clip(%237, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %239 = cast(%238, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %240 = annotation.stop_fusion(%239) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %241 = nn.conv2d(%240, meta[relay.Constant][52] /* ty=Tensor[(64, 384, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %242 = add(%241, meta[relay.Constant][53] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %243 = add(%242, 64 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %244 = right_shift(%243, 7 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %245 = clip(%244, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %246 = cast(%245, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %247 = annotation.stop_fusion(%246) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %248 = cast(%247, dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %249 = add(%221, %248) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %250 = clip(%249, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %251 = cast(%250, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %252 = annotation.stop_fusion(%251) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %253 = cast(%252, dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %254 = cast(%250, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %255 = annotation.stop_fusion(%254) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %256 = nn.conv2d(%255, meta[relay.Constant][54] /* ty=Tensor[(384, 64, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %257 = left_shift(%256, 19 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %258 = add(%257, meta[relay.Constant][55] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %259 = clip(%258, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %260 = add(%259, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %261 = right_shift(%260, 27 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %262 = clip(%261, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %263 = cast(%262, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %264 = annotation.stop_fusion(%263) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %265 = nn.conv2d(%264, meta[relay.Constant][56] /* ty=Tensor[(384, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %266 = add(%265, meta[relay.Constant][57] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %267 = clip(%266, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %268 = add(%267, 512 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %269 = right_shift(%268, 10 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %270 = clip(%269, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %271 = cast(%270, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %272 = annotation.stop_fusion(%271) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %273 = nn.conv2d(%272, meta[relay.Constant][58] /* ty=Tensor[(64, 384, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %274 = add(%273, meta[relay.Constant][59] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %275 = add(%274, 64 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %276 = right_shift(%275, 7 /* ty=int32 */) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %277 = clip(%276, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %278 = cast(%277, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %279 = annotation.stop_fusion(%278) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %280 = cast(%279, dtype="int32") /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %281 = add(%253, %280) /* ty=Tensor[(1, 64, 14, 14), int32] */;
  %282 = cast(%281, dtype="int8") /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %283 = annotation.stop_fusion(%282) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %284 = nn.conv2d(%283, meta[relay.Constant][60] /* ty=Tensor[(384, 64, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %285 = left_shift(%284, 19 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %286 = add(%285, meta[relay.Constant][61] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %287 = clip(%286, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %288 = add(%287, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %289 = right_shift(%288, 27 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %290 = clip(%289, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %291 = cast(%290, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %292 = annotation.stop_fusion(%291) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %293 = nn.conv2d(%292, meta[relay.Constant][62] /* ty=Tensor[(384, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %294 = add(%293, meta[relay.Constant][63] /* ty=Tensor[(384, 1, 1), int32] */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %295 = clip(%294, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %296 = add(%295, 512 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %297 = right_shift(%296, 10 /* ty=int32 */) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %298 = clip(%297, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 384, 14, 14), int32] */;
  %299 = cast(%298, dtype="int8") /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %300 = annotation.stop_fusion(%299) /* ty=Tensor[(1, 384, 14, 14), int8] */;
  %301 = nn.conv2d(%300, meta[relay.Constant][64] /* ty=Tensor[(96, 384, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %302 = add(%301, meta[relay.Constant][65] /* ty=Tensor[(96, 1, 1), int32] */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %303 = add(%302, 128 /* ty=int32 */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %304 = right_shift(%303, 8 /* ty=int32 */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %305 = clip(%304, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %306 = cast(%305, dtype="int8") /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %307 = annotation.stop_fusion(%306) /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %308 = cast(%307, dtype="int32") /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %309 = cast(%305, dtype="int8") /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %310 = annotation.stop_fusion(%309) /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %311 = nn.conv2d(%310, meta[relay.Constant][66] /* ty=Tensor[(576, 96, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %312 = left_shift(%311, 20 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %313 = add(%312, meta[relay.Constant][67] /* ty=Tensor[(576, 1, 1), int32] */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %314 = clip(%313, a_min=0f, a_max=2.57698e+10f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %315 = add(%314, 134217728 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %316 = right_shift(%315, 28 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %317 = clip(%316, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %318 = cast(%317, dtype="int8") /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %319 = annotation.stop_fusion(%318) /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %320 = nn.conv2d(%319, meta[relay.Constant][68] /* ty=Tensor[(576, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %321 = add(%320, meta[relay.Constant][69] /* ty=Tensor[(576, 1, 1), int32] */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %322 = clip(%321, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %323 = add(%322, 512 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %324 = right_shift(%323, 10 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %325 = clip(%324, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %326 = cast(%325, dtype="int8") /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %327 = annotation.stop_fusion(%326) /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %328 = nn.conv2d(%327, meta[relay.Constant][70] /* ty=Tensor[(96, 576, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %329 = add(%328, meta[relay.Constant][71] /* ty=Tensor[(96, 1, 1), int32] */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %330 = add(%329, 128 /* ty=int32 */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %331 = right_shift(%330, 8 /* ty=int32 */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %332 = clip(%331, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %333 = cast(%332, dtype="int8") /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %334 = annotation.stop_fusion(%333) /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %335 = cast(%334, dtype="int32") /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %336 = add(%308, %335) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %337 = clip(%336, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %338 = cast(%337, dtype="int8") /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %339 = annotation.stop_fusion(%338) /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %340 = cast(%339, dtype="int32") /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %341 = cast(%337, dtype="int8") /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %342 = annotation.stop_fusion(%341) /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %343 = nn.conv2d(%342, meta[relay.Constant][72] /* ty=Tensor[(576, 96, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %344 = left_shift(%343, 19 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %345 = add(%344, meta[relay.Constant][73] /* ty=Tensor[(576, 1, 1), int32] */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %346 = clip(%345, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %347 = add(%346, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %348 = right_shift(%347, 27 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %349 = clip(%348, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %350 = cast(%349, dtype="int8") /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %351 = annotation.stop_fusion(%350) /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %352 = nn.conv2d(%351, meta[relay.Constant][74] /* ty=Tensor[(576, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %353 = add(%352, meta[relay.Constant][75] /* ty=Tensor[(576, 1, 1), int32] */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %354 = clip(%353, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %355 = add(%354, 512 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %356 = right_shift(%355, 10 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %357 = clip(%356, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %358 = cast(%357, dtype="int8") /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %359 = annotation.stop_fusion(%358) /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %360 = nn.conv2d(%359, meta[relay.Constant][76] /* ty=Tensor[(96, 576, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %361 = add(%360, meta[relay.Constant][77] /* ty=Tensor[(96, 1, 1), int32] */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %362 = add(%361, 128 /* ty=int32 */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %363 = right_shift(%362, 8 /* ty=int32 */) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %364 = clip(%363, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %365 = cast(%364, dtype="int8") /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %366 = annotation.stop_fusion(%365) /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %367 = cast(%366, dtype="int32") /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %368 = add(%340, %367) /* ty=Tensor[(1, 96, 14, 14), int32] */;
  %369 = cast(%368, dtype="int8") /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %370 = annotation.stop_fusion(%369) /* ty=Tensor[(1, 96, 14, 14), int8] */;
  %371 = nn.conv2d(%370, meta[relay.Constant][78] /* ty=Tensor[(576, 96, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %372 = left_shift(%371, 19 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %373 = add(%372, meta[relay.Constant][79] /* ty=Tensor[(576, 1, 1), int32] */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %374 = clip(%373, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %375 = add(%374, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %376 = right_shift(%375, 27 /* ty=int32 */) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %377 = clip(%376, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 576, 14, 14), int32] */;
  %378 = cast(%377, dtype="int8") /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %379 = annotation.stop_fusion(%378) /* ty=Tensor[(1, 576, 14, 14), int8] */;
  %380 = nn.conv2d(%379, meta[relay.Constant][80] /* ty=Tensor[(576, 1, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 576, 7, 7), int32] */;
  %381 = add(%380, meta[relay.Constant][81] /* ty=Tensor[(576, 1, 1), int32] */) /* ty=Tensor[(1, 576, 7, 7), int32] */;
  %382 = clip(%381, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 576, 7, 7), int32] */;
  %383 = add(%382, 512 /* ty=int32 */) /* ty=Tensor[(1, 576, 7, 7), int32] */;
  %384 = right_shift(%383, 10 /* ty=int32 */) /* ty=Tensor[(1, 576, 7, 7), int32] */;
  %385 = clip(%384, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 576, 7, 7), int32] */;
  %386 = cast(%385, dtype="int8") /* ty=Tensor[(1, 576, 7, 7), int8] */;
  %387 = annotation.stop_fusion(%386) /* ty=Tensor[(1, 576, 7, 7), int8] */;
  %388 = nn.conv2d(%387, meta[relay.Constant][82] /* ty=Tensor[(160, 576, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %389 = add(%388, meta[relay.Constant][83] /* ty=Tensor[(160, 1, 1), int32] */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %390 = add(%389, 128 /* ty=int32 */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %391 = right_shift(%390, 8 /* ty=int32 */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %392 = clip(%391, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %393 = cast(%392, dtype="int8") /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %394 = annotation.stop_fusion(%393) /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %395 = cast(%394, dtype="int32") /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %396 = cast(%392, dtype="int8") /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %397 = annotation.stop_fusion(%396) /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %398 = nn.conv2d(%397, meta[relay.Constant][84] /* ty=Tensor[(960, 160, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %399 = left_shift(%398, 18 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %400 = add(%399, meta[relay.Constant][85] /* ty=Tensor[(960, 1, 1), int32] */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %401 = clip(%400, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %402 = add(%401, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %403 = right_shift(%402, 27 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %404 = clip(%403, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %405 = cast(%404, dtype="int8") /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %406 = annotation.stop_fusion(%405) /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %407 = nn.conv2d(%406, meta[relay.Constant][86] /* ty=Tensor[(960, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %408 = add(%407, meta[relay.Constant][87] /* ty=Tensor[(960, 1, 1), int32] */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %409 = clip(%408, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %410 = add(%409, 512 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %411 = right_shift(%410, 10 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %412 = clip(%411, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %413 = cast(%412, dtype="int8") /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %414 = annotation.stop_fusion(%413) /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %415 = nn.conv2d(%414, meta[relay.Constant][88] /* ty=Tensor[(160, 960, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %416 = add(%415, meta[relay.Constant][89] /* ty=Tensor[(160, 1, 1), int32] */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %417 = add(%416, 128 /* ty=int32 */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %418 = right_shift(%417, 8 /* ty=int32 */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %419 = clip(%418, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %420 = cast(%419, dtype="int8") /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %421 = annotation.stop_fusion(%420) /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %422 = cast(%421, dtype="int32") /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %423 = add(%395, %422) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %424 = clip(%423, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %425 = cast(%424, dtype="int8") /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %426 = annotation.stop_fusion(%425) /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %427 = cast(%426, dtype="int32") /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %428 = cast(%424, dtype="int8") /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %429 = annotation.stop_fusion(%428) /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %430 = nn.conv2d(%429, meta[relay.Constant][90] /* ty=Tensor[(960, 160, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %431 = left_shift(%430, 19 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %432 = add(%431, meta[relay.Constant][91] /* ty=Tensor[(960, 1, 1), int32] */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %433 = clip(%432, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %434 = add(%433, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %435 = right_shift(%434, 27 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %436 = clip(%435, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %437 = cast(%436, dtype="int8") /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %438 = annotation.stop_fusion(%437) /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %439 = nn.conv2d(%438, meta[relay.Constant][92] /* ty=Tensor[(960, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %440 = add(%439, meta[relay.Constant][93] /* ty=Tensor[(960, 1, 1), int32] */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %441 = clip(%440, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %442 = add(%441, 512 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %443 = right_shift(%442, 10 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %444 = clip(%443, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %445 = cast(%444, dtype="int8") /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %446 = annotation.stop_fusion(%445) /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %447 = nn.conv2d(%446, meta[relay.Constant][94] /* ty=Tensor[(160, 960, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %448 = add(%447, meta[relay.Constant][95] /* ty=Tensor[(160, 1, 1), int32] */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %449 = add(%448, 128 /* ty=int32 */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %450 = right_shift(%449, 8 /* ty=int32 */) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %451 = clip(%450, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %452 = cast(%451, dtype="int8") /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %453 = annotation.stop_fusion(%452) /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %454 = cast(%453, dtype="int32") /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %455 = add(%427, %454) /* ty=Tensor[(1, 160, 7, 7), int32] */;
  %456 = cast(%455, dtype="int8") /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %457 = annotation.stop_fusion(%456) /* ty=Tensor[(1, 160, 7, 7), int8] */;
  %458 = nn.conv2d(%457, meta[relay.Constant][96] /* ty=Tensor[(960, 160, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %459 = left_shift(%458, 18 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %460 = add(%459, meta[relay.Constant][97] /* ty=Tensor[(960, 1, 1), int32] */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %461 = clip(%460, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %462 = add(%461, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %463 = right_shift(%462, 27 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %464 = clip(%463, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %465 = cast(%464, dtype="int8") /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %466 = annotation.stop_fusion(%465) /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %467 = nn.conv2d(%466, meta[relay.Constant][98] /* ty=Tensor[(960, 1, 3, 3), int8] */, padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %468 = add(%467, meta[relay.Constant][99] /* ty=Tensor[(960, 1, 1), int32] */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %469 = clip(%468, a_min=0f, a_max=98304f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %470 = add(%469, 512 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %471 = right_shift(%470, 10 /* ty=int32 */) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %472 = clip(%471, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 960, 7, 7), int32] */;
  %473 = cast(%472, dtype="int8") /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %474 = annotation.stop_fusion(%473) /* ty=Tensor[(1, 960, 7, 7), int8] */;
  %475 = nn.conv2d(%474, meta[relay.Constant][100] /* ty=Tensor[(320, 960, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=320, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 320, 7, 7), int32] */;
  %476 = add(%475, meta[relay.Constant][101] /* ty=Tensor[(320, 1, 1), int32] */) /* ty=Tensor[(1, 320, 7, 7), int32] */;
  %477 = add(%476, 128 /* ty=int32 */) /* ty=Tensor[(1, 320, 7, 7), int32] */;
  %478 = right_shift(%477, 8 /* ty=int32 */) /* ty=Tensor[(1, 320, 7, 7), int32] */;
  %479 = clip(%478, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 320, 7, 7), int32] */;
  %480 = cast(%479, dtype="int8") /* ty=Tensor[(1, 320, 7, 7), int8] */;
  %481 = annotation.stop_fusion(%480) /* ty=Tensor[(1, 320, 7, 7), int8] */;
  %482 = nn.conv2d(%481, meta[relay.Constant][102] /* ty=Tensor[(1280, 320, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=1280, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 1280, 7, 7), int32] */;
  %483 = left_shift(%482, 18 /* ty=int32 */) /* ty=Tensor[(1, 1280, 7, 7), int32] */;
  %484 = add(%483, meta[relay.Constant][103] /* ty=Tensor[(1280, 1, 1), int32] */) /* ty=Tensor[(1, 1280, 7, 7), int32] */;
  %485 = clip(%484, a_min=0f, a_max=1.28849e+10f) /* ty=Tensor[(1, 1280, 7, 7), int32] */;
  %486 = add(%485, 67108864 /* ty=int32 */) /* ty=Tensor[(1, 1280, 7, 7), int32] */;
  %487 = right_shift(%486, 27 /* ty=int32 */) /* ty=Tensor[(1, 1280, 7, 7), int32] */;
  %488 = clip(%487, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 1280, 7, 7), int32] */;
  %489 = cast(%488, dtype="int8") /* ty=Tensor[(1, 1280, 7, 7), int8] */;
  %490 = annotation.stop_fusion(%489) /* ty=Tensor[(1, 1280, 7, 7), int8] */;
  %491 = cast(%490, dtype="float32") /* ty=Tensor[(1, 1280, 7, 7), float32] */;
  %492 = multiply(%491, 0.0625f /* ty=float32 */) /* ty=Tensor[(1, 1280, 7, 7), float32] */;
  %493 = nn.adaptive_avg_pool2d(%492, output_size=[1, 1]) /* ty=Tensor[(1, 1280, 1, 1), float32] */;
  %494 = reshape(%493, newshape=[1, -1]) /* ty=Tensor[(1, 1280), float32] */;
  %495 = nn.dense(%494, meta[relay.Constant][104] /* ty=Tensor[(1000, 1280), float32] */, units=None) /* ty=Tensor[(1, 1000), float32] */;
  add(%495, meta[relay.Constant][105] /* ty=Tensor[(1000), float32] */) /* ty=Tensor[(1, 1000), float32] */
}"""

network_wkls = []

conv_2d_count = 0
mp_count = 0

channels_re = re.compile(r"channels=(\d+)")
kernel_re = re.compile(r"kernel_size=\[(\d+), (?:\d+).*")
padding_re = re.compile(r"padding=\[(\d+), (?:\d+), (?:\d+), (?:\d+).*")
stride_re = re.compile(r"strides=\[(\d+), (?:\d+).*")
input_shape_re = re.compile(r"input0: Tensor\[\((?:\d+), (\d+), (\d+), (\d+)\).*")

ifm_shape_dict = {"height": 224, "width": 224, "channels": 3}
kernel_shape_dict = {"height": 0, "width": 0, "channels": 0, "stride": 0, "padding": 0}

output_file = "dataset/uart_sniffer/asp_dac/demo/mobilenet_v2_1x16x16_labels.txt"

def file_write_line_conv(cur_conv, layer_type):
    conv_out_height, conv_out_width, out_vol = calc_conv_output_size(cur_conv)

    ofm_dim_label = (conv_out_height, conv_out_width, cur_conv.out_filter)

    ifm_dim_label = (cur_conv.height, cur_conv.width, cur_conv.in_filter)

    kernel_dim_label = (cur_conv.hkernel, cur_conv.wkernel)

    stride_label = (cur_conv.hstride, cur_conv.wstride)

    pad_label = (cur_conv.hpad, cur_conv.wpad)

    return '\t'.join([layer_type, str(ifm_dim_label), str(ofm_dim_label), str(out_vol), str(kernel_dim_label),
                      str(stride_label), str(pad_label)]) + '\n'


def file_write_line_maxpool(cur_conv_cfg, maxpool_cfg, layer_type):
    conv_out_height, conv_out_width, _ = calc_conv_output_size(cur_conv_cfg)

    ifm_dim_label = (conv_out_height, conv_out_width, cur_conv_cfg.out_filter)

    mp_out_height, mp_out_width, out_vol = calc_maxpool_output_size(cur_conv_cfg, maxpool_cfg)

    ofm_dim_label = (mp_out_height, mp_out_height, cur_conv_cfg.out_filter)

    kernel_dim_label = (maxpool_cfg.hkernel, maxpool_cfg.wkernel)

    stride_label = (maxpool_cfg.hstride, maxpool_cfg.wstride)

    pad_label = (maxpool_cfg.hpad, maxpool_cfg.wpad)

    return '\t'.join([layer_type, str(ifm_dim_label), str(ofm_dim_label), str(out_vol), str(kernel_dim_label),
                      str(stride_label), str(pad_label)]) + '\n'

def generate_labels_file(network_wkls, output_file):
    layers_nt = network_wkls
    with open(output_file, 'w+') as myfile:
        myfile.write(
            '\t'.join(["layer_type", "ifm_dim", "ofm_dim", "output_vol", "kernel_dim", "stride", "pad"]) + '\n')



    layer_type = []
    cur_conv = None
    for i, layer in enumerate(layers_nt):

        if isinstance(layer, Conv2DWorkload):
            if i > 0 and len(layer_type) > 0:
                with open(output_file, 'a') as myfile:
                    myfile.write(file_write_line_conv(cur_conv, "".join(layer_type)))
            layer_type = []
            cur_conv = layer
            layer_type.append('C')
        elif isinstance(layer, BatchNorm2DConfig):
            layer_type.append('B')
        elif isinstance(layer, ReluConfig):
            layer_type.append('R')
        elif isinstance(layer, MaxPool2DConfig):
            if i > 0 and len(layer_type) > 0:
                with open(output_file, 'a') as myfile:
                    myfile.write(file_write_line_conv(cur_conv, "".join(layer_type)))

            layer_type = ['M']
            with open(output_file, 'a') as myfile:
                myfile.write(file_write_line_maxpool(cur_conv, layer, "".join(layer_type)))

            layer_type = []

    if len(layer_type) > 0:
        with open(output_file, 'a') as myfile:
            myfile.write(file_write_line_conv(cur_conv, "".join(layer_type)))


for line in mod_string.split("\n"):
    # get input shape
    match = input_shape_re.search(line)
    if match:
        input_shape = tuple(map(int, match.groups()))
        ifm_shape_dict["height"] = input_shape[1]
        ifm_shape_dict["width"] = input_shape[2]
        ifm_shape_dict["channels"] = input_shape[0]
        continue

    # get convolution dimensions
    if "nn.conv2d" in line:
        conv_2d_count += 1
        match = channels_re.search(line)
        kernel_shape_dict["channels"] = int(match.group(1))

        match = kernel_re.search(line)
        kernel_shape_dict["height"] = int(match.group(1))
        kernel_shape_dict["width"] = int(match.group(1))

        match = padding_re.search(line)
        if match:
            kernel_shape_dict["padding"] = int(match.group(1))
        else:
            kernel_shape_dict["padding"] = 0

        match = stride_re.search(line)
        if match:
            kernel_shape_dict["stride"] = int(match.group(1))
        else:
            kernel_shape_dict["stride"] = 1

        # network_wkls.append(kernel_shape_dict.copy())

        conv_wkl = Conv2DWorkload(batch=env.BATCH, height=ifm_shape_dict["height"], width=ifm_shape_dict["width"], in_filter=ifm_shape_dict["channels"],
                                  out_filter=kernel_shape_dict["channels"], hkernel=kernel_shape_dict["height"],
                                  wkernel=kernel_shape_dict["width"], hstride=kernel_shape_dict["stride"], wstride=kernel_shape_dict["stride"],
                                  hpad=kernel_shape_dict["padding"], wpad=kernel_shape_dict["padding"])

        conv_out_height, conv_out_width, _ = calc_conv_output_size(conv_wkl)
        ifm_shape_dict["height"] = conv_out_height
        ifm_shape_dict["width"] = conv_out_width
        ifm_shape_dict["channels"] = conv_wkl.out_filter

        if conv_2d_count >= 2:
            network_wkls.append(conv_wkl)

        continue

print(network_wkls)
generate_labels_file(network_wkls, output_file)



