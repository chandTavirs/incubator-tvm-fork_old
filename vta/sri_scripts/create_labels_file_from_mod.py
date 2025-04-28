import argparse
import os
import re

from alt_wkl_configs import *
import ast
from get_output_sizes import calc_conv_output_size, calc_maxpool_output_size, calc_conv_input_size
from vta import environment
env = environment.get_env()

mod_string = """
#[version = "0.0.5"]
def @main(%input0: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 10), float32] {
  %0 = reshape(%input0, newshape=[-1, 3, 224, 224]) /* ty=Tensor[(1, 3, 224, 224), float32] */;
  %1 = nn.conv2d(%0, meta[relay.Constant][0] /* ty=Tensor[(64, 3, 7, 7), float32] */, strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7]) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %2 = add(%1, meta[relay.Constant][1] /* ty=Tensor[(64, 1, 1), float32] */) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %3 = annotation.stop_fusion(%2) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %4 = multiply(%3, meta[relay.Constant][2] /* ty=Tensor[(64, 1, 1), float32] */) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %5 = add(%4, meta[relay.Constant][3] /* ty=Tensor[(64, 1, 1), float32] */) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %6 = nn.relu(%5) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %7 = annotation.stop_fusion(%6) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %8 = multiply(%7, 16f /* ty=float32 */) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %9 = round(%8) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %10 = clip(%9, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %11 = cast(%10, dtype="int8") /* ty=Tensor[(1, 64, 112, 112), int8] */;
  %12 = nn.conv2d(%11, meta[relay.Constant][4] /* ty=Tensor[(64, 64, 7, 7), int8] */, padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7], out_dtype="int32") /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %13 = add(%12, meta[relay.Constant][5] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %14 = add(%13, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %15 = right_shift(%14, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %16 = clip(%15, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %17 = cast(%16, dtype="int8") /* ty=Tensor[(1, 64, 112, 112), int8] */;
  %18 = annotation.stop_fusion(%17) /* ty=Tensor[(1, 64, 112, 112), int8] */;
  %19 = cast(%18, dtype="int32") /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %20 = annotation.stop_fusion(%6) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %21 = multiply(%20, 16f /* ty=float32 */) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %22 = round(%21) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %23 = clip(%22, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 112, 112), float32] */;
  %24 = cast(%23, dtype="int32") /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %25 = annotation.stop_fusion(%24) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %26 = add(%19, %25) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %27 = nn.relu(%26) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %28 = clip(%27, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 112, 112), int32] */;
  %29 = cast(%28, dtype="int8") /* ty=Tensor[(1, 64, 112, 112), int8] */;
  %30 = nn.max_pool2d(%29, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0]) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %31 = clip(%30, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %32 = cast(%31, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %33 = annotation.stop_fusion(%32) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %34 = clip(%33, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %35 = strided_slice(%34, begin=[0, 0, 0, 0], end=[1, 16, 56, 56], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %36 = nn.conv2d(%35, meta[relay.Constant][6] /* ty=Tensor[(64, 16, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %37 = add(%36, meta[relay.Constant][7] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %38 = add(%37, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %39 = right_shift(%38, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %40 = clip(%39, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %41 = cast(%40, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %42 = annotation.stop_fusion(%41) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %43 = cast(%42, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %44 = strided_slice(%34, begin=[0, 16, 0, 0], end=[1, 32, 56, 56], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %45 = nn.conv2d(%44, meta[relay.Constant][8] /* ty=Tensor[(64, 16, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %46 = add(%45, meta[relay.Constant][9] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %47 = add(%46, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %48 = right_shift(%47, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %49 = clip(%48, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %50 = cast(%49, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %51 = annotation.stop_fusion(%50) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %52 = cast(%51, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %53 = add(%43, %52) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %54 = cast(%53, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %55 = annotation.stop_fusion(%54) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %56 = cast(%55, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %57 = strided_slice(%34, begin=[0, 32, 0, 0], end=[1, 48, 56, 56], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %58 = nn.conv2d(%57, meta[relay.Constant][10] /* ty=Tensor[(64, 16, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %59 = add(%58, meta[relay.Constant][11] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %60 = add(%59, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %61 = right_shift(%60, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %62 = clip(%61, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %63 = cast(%62, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %64 = annotation.stop_fusion(%63) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %65 = cast(%64, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %66 = add(%56, %65) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %67 = cast(%66, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %68 = annotation.stop_fusion(%67) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %69 = cast(%68, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %70 = strided_slice(%34, begin=[0, 48, 0, 0], end=[1, 64, 56, 56], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %71 = nn.conv2d(%70, meta[relay.Constant][12] /* ty=Tensor[(64, 16, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %72 = add(%71, meta[relay.Constant][13] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %73 = add(%72, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %74 = right_shift(%73, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %75 = clip(%74, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %76 = cast(%75, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %77 = annotation.stop_fusion(%76) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %78 = cast(%77, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %79 = add(%69, %78) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %80 = cast(%79, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %81 = annotation.stop_fusion(%80) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %82 = cast(%81, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %83 = multiply(%82, meta[relay.Constant][14] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %84 = add(%83, meta[relay.Constant][15] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %85 = nn.relu(%84) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %86 = add(%85, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %87 = right_shift(%86, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %88 = clip(%87, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %89 = cast(%88, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %90 = annotation.stop_fusion(%89) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %91 = nn.conv2d(%90, meta[relay.Constant][16] /* ty=Tensor[(64, 64, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %92 = add(%91, meta[relay.Constant][17] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %93 = add(%92, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %94 = right_shift(%93, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %95 = clip(%94, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %96 = cast(%95, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %97 = annotation.stop_fusion(%96) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %98 = cast(%97, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %99 = multiply(%98, meta[relay.Constant][18] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %100 = add(%99, meta[relay.Constant][19] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %101 = add(%100, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %102 = right_shift(%101, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %103 = clip(%102, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %104 = cast(%103, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %105 = annotation.stop_fusion(%104) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %106 = cast(%105, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %107 = cast(%31, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %108 = annotation.stop_fusion(%107) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %109 = cast(%108, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %110 = add(%106, %109) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %111 = nn.relu(%110) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %112 = clip(%111, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %113 = cast(%112, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %114 = annotation.stop_fusion(%113) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %115 = nn.conv2d(%114, meta[relay.Constant][20] /* ty=Tensor[(16, 64, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %116 = add(%115, meta[relay.Constant][21] /* ty=Tensor[(16, 1, 1), int32] */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %117 = add(%116, 16384 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %118 = right_shift(%117, 15 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %119 = clip(%118, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %120 = cast(%119, dtype="int8") /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %121 = annotation.stop_fusion(%120) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %122 = cast(%121, dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %123 = left_shift(%122, 24 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %124 = cast(%112, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %125 = annotation.stop_fusion(%124) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %126 = nn.conv2d(%125, meta[relay.Constant][22] /* ty=Tensor[(16, 64, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %127 = add(%126, meta[relay.Constant][23] /* ty=Tensor[(16, 1, 1), int32] */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %128 = add(%127, 16384 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %129 = right_shift(%128, 15 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %130 = clip(%129, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %131 = cast(%130, dtype="int8") /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %132 = annotation.stop_fusion(%131) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %133 = cast(%132, dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %134 = left_shift(%133, 24 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %135 = cast(%112, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %136 = annotation.stop_fusion(%135) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %137 = nn.conv2d(%136, meta[relay.Constant][24] /* ty=Tensor[(16, 64, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %138 = add(%137, meta[relay.Constant][25] /* ty=Tensor[(16, 1, 1), int32] */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %139 = add(%138, 16384 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %140 = right_shift(%139, 15 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %141 = clip(%140, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %142 = cast(%141, dtype="int8") /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %143 = annotation.stop_fusion(%142) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %144 = cast(%143, dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %145 = left_shift(%144, 24 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %146 = cast(%112, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %147 = annotation.stop_fusion(%146) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %148 = nn.conv2d(%147, meta[relay.Constant][26] /* ty=Tensor[(16, 64, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=16, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %149 = add(%148, meta[relay.Constant][27] /* ty=Tensor[(16, 1, 1), int32] */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %150 = add(%149, 16384 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %151 = right_shift(%150, 15 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %152 = clip(%151, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %153 = cast(%152, dtype="int8") /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %154 = annotation.stop_fusion(%153) /* ty=Tensor[(1, 16, 56, 56), int8] */;
  %155 = cast(%154, dtype="int32") /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %156 = left_shift(%155, 24 /* ty=int32 */) /* ty=Tensor[(1, 16, 56, 56), int32] */;
  %157 = (%123, %134, %145, %156);
  %158 = concatenate(%157, axis=1) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %159 = add(%158, 8388608 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %160 = right_shift(%159, 24 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %161 = clip(%160, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %162 = multiply(%161, meta[relay.Constant][28] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %163 = add(%162, meta[relay.Constant][29] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %164 = nn.relu(%163) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %165 = add(%164, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %166 = right_shift(%165, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %167 = clip(%166, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %168 = cast(%167, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %169 = nn.conv2d(%168, meta[relay.Constant][30] /* ty=Tensor[(64, 64, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %170 = add(%169, meta[relay.Constant][31] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %171 = add(%170, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %172 = right_shift(%171, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %173 = clip(%172, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %174 = cast(%173, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %175 = annotation.stop_fusion(%174) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %176 = cast(%175, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %177 = multiply(%176, meta[relay.Constant][32] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %178 = add(%177, meta[relay.Constant][33] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %179 = add(%178, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %180 = right_shift(%179, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %181 = clip(%180, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %182 = cast(%181, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %183 = annotation.stop_fusion(%182) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %184 = cast(%183, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %185 = cast(%112, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %186 = annotation.stop_fusion(%185) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %187 = cast(%186, dtype="int32") /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %188 = add(%184, %187) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %189 = nn.relu(%188) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %190 = clip(%189, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 56, 56), int32] */;
  %191 = cast(%190, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %192 = annotation.stop_fusion(%191) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %193 = nn.conv2d(%192, meta[relay.Constant][34] /* ty=Tensor[(64, 64, 1, 1), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %194 = add(%193, meta[relay.Constant][35] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %195 = add(%194, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %196 = right_shift(%195, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %197 = clip(%196, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %198 = cast(%197, dtype="int8") /* ty=Tensor[(1, 64, 28, 28), int8] */;
  %199 = annotation.stop_fusion(%198) /* ty=Tensor[(1, 64, 28, 28), int8] */;
  %200 = cast(%190, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %201 = annotation.stop_fusion(%200) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %202 = nn.conv2d(%201, meta[relay.Constant][36] /* ty=Tensor[(64, 64, 1, 1), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %203 = add(%202, meta[relay.Constant][37] /* ty=Tensor[(64, 1, 1), int32] */) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %204 = add(%203, 16384 /* ty=int32 */) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %205 = right_shift(%204, 15 /* ty=int32 */) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %206 = clip(%205, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %207 = cast(%206, dtype="int8") /* ty=Tensor[(1, 64, 28, 28), int8] */;
  %208 = annotation.stop_fusion(%207) /* ty=Tensor[(1, 64, 28, 28), int8] */;
  %209 = (%199, %208);
  %210 = concatenate(%209, axis=1) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %211 = clip(%210, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %212 = cast(%211, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %213 = multiply(%212, meta[relay.Constant][38] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %214 = add(%213, meta[relay.Constant][39] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %215 = cast(%190, dtype="int8") /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %216 = annotation.stop_fusion(%215) /* ty=Tensor[(1, 64, 56, 56), int8] */;
  %217 = nn.conv2d(%216, meta[relay.Constant][40] /* ty=Tensor[(128, 64, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %218 = add(%217, meta[relay.Constant][41] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %219 = add(%218, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %220 = right_shift(%219, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %221 = clip(%220, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %222 = cast(%221, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %223 = annotation.stop_fusion(%222) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %224 = cast(%223, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %225 = multiply(%224, meta[relay.Constant][42] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %226 = add(%225, meta[relay.Constant][43] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %227 = nn.relu(%226) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %228 = add(%227, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %229 = right_shift(%228, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %230 = clip(%229, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %231 = cast(%230, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %232 = annotation.stop_fusion(%231) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %233 = nn.conv2d(%232, meta[relay.Constant][44] /* ty=Tensor[(128, 128, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %234 = add(%233, meta[relay.Constant][45] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %235 = add(%234, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %236 = right_shift(%235, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %237 = clip(%236, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %238 = cast(%237, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %239 = annotation.stop_fusion(%238) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %240 = cast(%239, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %241 = cast(%230, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %242 = annotation.stop_fusion(%241) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %243 = cast(%242, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %244 = add(%240, %243) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %245 = nn.relu(%244) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %246 = cast(%245, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %247 = annotation.stop_fusion(%246) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %248 = nn.conv2d(%247, meta[relay.Constant][46] /* ty=Tensor[(128, 128, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %249 = add(%248, meta[relay.Constant][47] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %250 = add(%249, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %251 = right_shift(%250, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %252 = clip(%251, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %253 = cast(%252, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %254 = annotation.stop_fusion(%253) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %255 = cast(%254, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %256 = multiply(%255, meta[relay.Constant][48] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %257 = add(%256, meta[relay.Constant][49] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %258 = add(%257, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %259 = right_shift(%258, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %260 = clip(%259, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %261 = cast(%260, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %262 = annotation.stop_fusion(%261) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %263 = cast(%262, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %264 = left_shift(%263, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %265 = add(%214, %264) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %266 = nn.relu(%265) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %267 = add(%266, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %268 = right_shift(%267, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %269 = clip(%268, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %270 = cast(%269, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %271 = nn.conv2d(%270, meta[relay.Constant][50] /* ty=Tensor[(128, 128, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %272 = add(%271, meta[relay.Constant][51] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %273 = add(%272, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %274 = right_shift(%273, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %275 = clip(%274, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %276 = cast(%275, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %277 = annotation.stop_fusion(%276) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %278 = cast(%277, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %279 = multiply(%278, meta[relay.Constant][52] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %280 = add(%279, meta[relay.Constant][53] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %281 = nn.relu(%280) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %282 = add(%281, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %283 = right_shift(%282, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %284 = clip(%283, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %285 = cast(%284, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %286 = annotation.stop_fusion(%285) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %287 = clip(%286, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %288 = strided_slice(%287, begin=[0, 0, 0, 0], end=[1, 32, 28, 28], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %289 = nn.conv2d(%288, meta[relay.Constant][54] /* ty=Tensor[(128, 32, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %290 = add(%289, meta[relay.Constant][55] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %291 = add(%290, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %292 = right_shift(%291, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %293 = clip(%292, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %294 = cast(%293, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %295 = annotation.stop_fusion(%294) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %296 = cast(%295, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %297 = strided_slice(%287, begin=[0, 32, 0, 0], end=[1, 64, 28, 28], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %298 = nn.conv2d(%297, meta[relay.Constant][56] /* ty=Tensor[(128, 32, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %299 = add(%298, meta[relay.Constant][57] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %300 = add(%299, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %301 = right_shift(%300, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %302 = clip(%301, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %303 = cast(%302, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %304 = annotation.stop_fusion(%303) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %305 = cast(%304, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %306 = add(%296, %305) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %307 = cast(%306, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %308 = annotation.stop_fusion(%307) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %309 = cast(%308, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %310 = strided_slice(%287, begin=[0, 64, 0, 0], end=[1, 96, 28, 28], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %311 = nn.conv2d(%310, meta[relay.Constant][58] /* ty=Tensor[(128, 32, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %312 = add(%311, meta[relay.Constant][59] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %313 = add(%312, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %314 = right_shift(%313, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %315 = clip(%314, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %316 = cast(%315, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %317 = annotation.stop_fusion(%316) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %318 = cast(%317, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %319 = add(%309, %318) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %320 = cast(%319, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %321 = annotation.stop_fusion(%320) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %322 = cast(%321, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %323 = strided_slice(%287, begin=[0, 96, 0, 0], end=[1, 128, 28, 28], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 32, 28, 28), int8] */;
  %324 = nn.conv2d(%323, meta[relay.Constant][60] /* ty=Tensor[(128, 32, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %325 = add(%324, meta[relay.Constant][61] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %326 = add(%325, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %327 = right_shift(%326, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %328 = clip(%327, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %329 = cast(%328, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %330 = annotation.stop_fusion(%329) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %331 = cast(%330, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %332 = add(%322, %331) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %333 = cast(%332, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %334 = annotation.stop_fusion(%333) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %335 = cast(%334, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %336 = multiply(%335, meta[relay.Constant][62] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %337 = add(%336, meta[relay.Constant][63] /* ty=Tensor[(128, 1, 1), int32] */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %338 = add(%337, 16384 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %339 = right_shift(%338, 15 /* ty=int32 */) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %340 = clip(%339, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %341 = cast(%340, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %342 = annotation.stop_fusion(%341) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %343 = cast(%342, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %344 = cast(%269, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %345 = annotation.stop_fusion(%344) /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %346 = cast(%345, dtype="int32") /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %347 = add(%343, %346) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %348 = nn.relu(%347) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %349 = clip(%348, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 128, 28, 28), int32] */;
  %350 = cast(%349, dtype="int8") /* ty=Tensor[(1, 128, 28, 28), int8] */;
  %351 = nn.conv2d(%350, meta[relay.Constant][64] /* ty=Tensor[(256, 128, 1, 1), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %352 = add(%351, meta[relay.Constant][65] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %353 = add(%352, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %354 = right_shift(%353, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %355 = clip(%354, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %356 = cast(%355, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %357 = annotation.stop_fusion(%356) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %358 = cast(%357, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %359 = multiply(%358, meta[relay.Constant][66] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %360 = add(%359, meta[relay.Constant][67] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %361 = add(%360, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %362 = right_shift(%361, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %363 = clip(%362, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %364 = cast(%363, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %365 = annotation.stop_fusion(%364) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %366 = cast(%365, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %367 = strided_slice(%349, begin=[0, 0, 0, 0], end=[1, 64, 28, 28], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %368 = cast(%367, dtype="int8") /* ty=Tensor[(1, 64, 28, 28), int8] */;
  %369 = nn.conv2d(%368, meta[relay.Constant][68] /* ty=Tensor[(256, 64, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %370 = add(%369, meta[relay.Constant][69] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %371 = add(%370, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %372 = right_shift(%371, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %373 = clip(%372, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %374 = cast(%373, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %375 = annotation.stop_fusion(%374) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %376 = cast(%375, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %377 = strided_slice(%349, begin=[0, 64, 0, 0], end=[1, 128, 28, 28], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 64, 28, 28), int32] */;
  %378 = cast(%377, dtype="int8") /* ty=Tensor[(1, 64, 28, 28), int8] */;
  %379 = nn.conv2d(%378, meta[relay.Constant][70] /* ty=Tensor[(256, 64, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %380 = add(%379, meta[relay.Constant][71] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %381 = add(%380, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %382 = right_shift(%381, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %383 = clip(%382, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %384 = cast(%383, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %385 = annotation.stop_fusion(%384) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %386 = cast(%385, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %387 = add(%376, %386) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %388 = cast(%387, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %389 = annotation.stop_fusion(%388) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %390 = cast(%389, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %391 = multiply(%390, meta[relay.Constant][72] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %392 = add(%391, meta[relay.Constant][73] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %393 = nn.relu(%392) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %394 = add(%393, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %395 = right_shift(%394, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %396 = clip(%395, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %397 = cast(%396, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %398 = annotation.stop_fusion(%397) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %399 = nn.conv2d(%398, meta[relay.Constant][74] /* ty=Tensor[(256, 256, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %400 = add(%399, meta[relay.Constant][75] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %401 = add(%400, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %402 = right_shift(%401, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %403 = clip(%402, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %404 = cast(%403, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %405 = annotation.stop_fusion(%404) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %406 = cast(%405, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %407 = cast(%396, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %408 = annotation.stop_fusion(%407) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %409 = cast(%408, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %410 = add(%406, %409) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %411 = nn.relu(%410) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %412 = cast(%411, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %413 = annotation.stop_fusion(%412) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %414 = nn.conv2d(%413, meta[relay.Constant][76] /* ty=Tensor[(256, 256, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %415 = add(%414, meta[relay.Constant][77] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %416 = add(%415, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %417 = right_shift(%416, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %418 = clip(%417, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %419 = cast(%418, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %420 = annotation.stop_fusion(%419) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %421 = cast(%420, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %422 = multiply(%421, meta[relay.Constant][78] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %423 = add(%422, meta[relay.Constant][79] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %424 = add(%423, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %425 = right_shift(%424, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %426 = clip(%425, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %427 = cast(%426, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %428 = annotation.stop_fusion(%427) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %429 = cast(%428, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %430 = add(%366, %429) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %431 = nn.relu(%430) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %432 = clip(%431, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %433 = cast(%432, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %434 = annotation.stop_fusion(%433) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %435 = nn.conv2d(%434, meta[relay.Constant][80] /* ty=Tensor[(256, 256, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %436 = add(%435, meta[relay.Constant][81] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %437 = add(%436, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %438 = right_shift(%437, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %439 = clip(%438, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %440 = cast(%439, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %441 = annotation.stop_fusion(%440) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %442 = cast(%441, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %443 = cast(%432, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %444 = annotation.stop_fusion(%443) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %445 = cast(%444, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %446 = add(%442, %445) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %447 = nn.relu(%446) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %448 = clip(%447, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %449 = cast(%448, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %450 = annotation.stop_fusion(%449) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %451 = nn.conv2d(%450, meta[relay.Constant][82] /* ty=Tensor[(256, 256, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %452 = add(%451, meta[relay.Constant][83] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %453 = add(%452, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %454 = right_shift(%453, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %455 = clip(%454, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %456 = cast(%455, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %457 = annotation.stop_fusion(%456) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %458 = cast(%457, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %459 = multiply(%458, meta[relay.Constant][84] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %460 = add(%459, meta[relay.Constant][85] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %461 = nn.relu(%460) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %462 = add(%461, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %463 = right_shift(%462, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %464 = clip(%463, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %465 = cast(%464, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %466 = annotation.stop_fusion(%465) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %467 = nn.conv2d(%466, meta[relay.Constant][86] /* ty=Tensor[(256, 256, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %468 = add(%467, meta[relay.Constant][87] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %469 = nn.relu(%468) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %470 = add(%469, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %471 = right_shift(%470, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %472 = clip(%471, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %473 = cast(%472, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %474 = annotation.stop_fusion(%473) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %475 = clip(%474, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %476 = strided_slice(%475, begin=[0, 0, 0, 0], end=[1, 128, 14, 14], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 128, 14, 14), int8] */;
  %477 = nn.conv2d(%476, meta[relay.Constant][88] /* ty=Tensor[(256, 128, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %478 = add(%477, meta[relay.Constant][89] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %479 = add(%478, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %480 = right_shift(%479, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %481 = clip(%480, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %482 = cast(%481, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %483 = annotation.stop_fusion(%482) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %484 = cast(%483, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %485 = strided_slice(%475, begin=[0, 128, 0, 0], end=[1, 256, 14, 14], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 128, 14, 14), int8] */;
  %486 = nn.conv2d(%485, meta[relay.Constant][90] /* ty=Tensor[(256, 128, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %487 = add(%486, meta[relay.Constant][91] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %488 = add(%487, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %489 = right_shift(%488, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %490 = clip(%489, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %491 = cast(%490, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %492 = annotation.stop_fusion(%491) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %493 = cast(%492, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %494 = add(%484, %493) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %495 = cast(%494, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %496 = annotation.stop_fusion(%495) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %497 = cast(%496, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %498 = multiply(%497, meta[relay.Constant][92] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %499 = add(%498, meta[relay.Constant][93] /* ty=Tensor[(256, 1, 1), int32] */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %500 = add(%499, 16384 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %501 = right_shift(%500, 15 /* ty=int32 */) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %502 = clip(%501, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %503 = cast(%502, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %504 = annotation.stop_fusion(%503) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %505 = cast(%504, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %506 = cast(%448, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %507 = annotation.stop_fusion(%506) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %508 = cast(%507, dtype="int32") /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %509 = add(%505, %508) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %510 = nn.relu(%509) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %511 = clip(%510, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int32] */;
  %512 = cast(%511, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %513 = annotation.stop_fusion(%512) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %514 = clip(%513, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %515 = strided_slice(%514, begin=[0, 0, 0, 0], end=[1, 64, 14, 14], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %516 = nn.conv2d(%515, meta[relay.Constant][94] /* ty=Tensor[(512, 64, 1, 1), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %517 = add(%516, meta[relay.Constant][95] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %518 = add(%517, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %519 = right_shift(%518, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %520 = clip(%519, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %521 = cast(%520, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %522 = annotation.stop_fusion(%521) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %523 = cast(%522, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %524 = strided_slice(%514, begin=[0, 64, 0, 0], end=[1, 128, 14, 14], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %525 = nn.conv2d(%524, meta[relay.Constant][96] /* ty=Tensor[(512, 64, 1, 1), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %526 = add(%525, meta[relay.Constant][97] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %527 = add(%526, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %528 = right_shift(%527, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %529 = clip(%528, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %530 = cast(%529, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %531 = annotation.stop_fusion(%530) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %532 = cast(%531, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %533 = add(%523, %532) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %534 = cast(%533, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %535 = annotation.stop_fusion(%534) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %536 = cast(%535, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %537 = strided_slice(%514, begin=[0, 128, 0, 0], end=[1, 192, 14, 14], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %538 = nn.conv2d(%537, meta[relay.Constant][98] /* ty=Tensor[(512, 64, 1, 1), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %539 = add(%538, meta[relay.Constant][99] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %540 = add(%539, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %541 = right_shift(%540, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %542 = clip(%541, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %543 = cast(%542, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %544 = annotation.stop_fusion(%543) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %545 = cast(%544, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %546 = add(%536, %545) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %547 = cast(%546, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %548 = annotation.stop_fusion(%547) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %549 = cast(%548, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %550 = strided_slice(%514, begin=[0, 192, 0, 0], end=[1, 256, 14, 14], strides=[1, 1, 1, 1], axes=None) /* ty=Tensor[(1, 64, 14, 14), int8] */;
  %551 = nn.conv2d(%550, meta[relay.Constant][100] /* ty=Tensor[(512, 64, 1, 1), int8] */, strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %552 = add(%551, meta[relay.Constant][101] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %553 = add(%552, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %554 = right_shift(%553, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %555 = clip(%554, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %556 = cast(%555, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %557 = annotation.stop_fusion(%556) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %558 = cast(%557, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %559 = add(%549, %558) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %560 = cast(%559, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %561 = annotation.stop_fusion(%560) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %562 = cast(%561, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %563 = multiply(%562, meta[relay.Constant][102] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %564 = add(%563, meta[relay.Constant][103] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %565 = add(%564, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %566 = right_shift(%565, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %567 = clip(%566, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %568 = cast(%567, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %569 = annotation.stop_fusion(%568) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %570 = cast(%569, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %571 = cast(%511, dtype="int8") /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %572 = annotation.stop_fusion(%571) /* ty=Tensor[(1, 256, 14, 14), int8] */;
  %573 = nn.conv2d(%572, meta[relay.Constant][104] /* ty=Tensor[(512, 256, 3, 3), int8] */, strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %574 = add(%573, meta[relay.Constant][105] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %575 = add(%574, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %576 = right_shift(%575, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %577 = clip(%576, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %578 = cast(%577, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %579 = annotation.stop_fusion(%578) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %580 = cast(%579, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %581 = multiply(%580, meta[relay.Constant][106] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %582 = add(%581, meta[relay.Constant][107] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %583 = nn.relu(%582) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %584 = add(%583, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %585 = right_shift(%584, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %586 = clip(%585, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %587 = cast(%586, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %588 = annotation.stop_fusion(%587) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %589 = nn.conv2d(%588, meta[relay.Constant][108] /* ty=Tensor[(512, 512, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %590 = add(%589, meta[relay.Constant][109] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %591 = nn.relu(%590) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %592 = add(%591, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %593 = right_shift(%592, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %594 = clip(%593, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %595 = cast(%594, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %596 = annotation.stop_fusion(%595) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %597 = nn.conv2d(%596, meta[relay.Constant][110] /* ty=Tensor[(512, 512, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %598 = add(%597, meta[relay.Constant][111] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %599 = add(%598, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %600 = right_shift(%599, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %601 = clip(%600, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %602 = cast(%601, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %603 = annotation.stop_fusion(%602) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %604 = cast(%603, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %605 = multiply(%604, meta[relay.Constant][112] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %606 = add(%605, meta[relay.Constant][113] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %607 = add(%606, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %608 = right_shift(%607, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %609 = clip(%608, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %610 = cast(%609, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %611 = annotation.stop_fusion(%610) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %612 = cast(%611, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %613 = add(%570, %612) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %614 = nn.relu(%613) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %615 = clip(%614, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %616 = cast(%615, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %617 = annotation.stop_fusion(%616) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %618 = nn.conv2d(%617, meta[relay.Constant][114] /* ty=Tensor[(512, 512, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %619 = add(%618, meta[relay.Constant][115] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %620 = add(%619, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %621 = right_shift(%620, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %622 = clip(%621, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %623 = cast(%622, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %624 = annotation.stop_fusion(%623) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %625 = cast(%624, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %626 = multiply(%625, meta[relay.Constant][116] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %627 = add(%626, meta[relay.Constant][117] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %628 = nn.relu(%627) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %629 = add(%628, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %630 = right_shift(%629, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %631 = clip(%630, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %632 = cast(%631, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %633 = annotation.stop_fusion(%632) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %634 = nn.conv2d(%633, meta[relay.Constant][118] /* ty=Tensor[(512, 512, 3, 3), int8] */, padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %635 = add(%634, meta[relay.Constant][119] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %636 = add(%635, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %637 = right_shift(%636, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %638 = clip(%637, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %639 = cast(%638, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %640 = annotation.stop_fusion(%639) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %641 = cast(%640, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %642 = multiply(%641, meta[relay.Constant][120] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %643 = add(%642, meta[relay.Constant][121] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %644 = add(%643, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %645 = right_shift(%644, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %646 = clip(%645, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %647 = cast(%646, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %648 = annotation.stop_fusion(%647) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %649 = cast(%648, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %650 = cast(%615, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %651 = annotation.stop_fusion(%650) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %652 = cast(%651, dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %653 = add(%649, %652) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %654 = nn.relu(%653) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %655 = cast(%654, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %656 = annotation.stop_fusion(%655) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %657 = nn.conv2d(%656, meta[relay.Constant][122] /* ty=Tensor[(512, 512, 1, 1), int8] */, padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1], out_dtype="int32") /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %658 = add(%657, meta[relay.Constant][123] /* ty=Tensor[(512, 1, 1), int32] */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %659 = nn.relu(%658) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %660 = add(%659, 16384 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %661 = right_shift(%660, 15 /* ty=int32 */) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %662 = clip(%661, a_min=-127f, a_max=127f) /* ty=Tensor[(1, 512, 7, 7), int32] */;
  %663 = cast(%662, dtype="int8") /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %664 = annotation.stop_fusion(%663) /* ty=Tensor[(1, 512, 7, 7), int8] */;
  %665 = cast(%664, dtype="float32") /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %666 = multiply(%665, 0.0625f /* ty=float32 */) /* ty=Tensor[(1, 512, 7, 7), float32] */;
  %667 = nn.adaptive_avg_pool2d(%666, output_size=[1, 1]) /* ty=Tensor[(1, 512, 1, 1), float32] */;
  %668 = reshape(%667, newshape=[-1, 512]) /* ty=Tensor[(1, 512), float32] */;
  %669 = nn.dense(%668, meta[relay.Constant][124] /* ty=Tensor[(10, 512), float32] */, units=None) /* ty=Tensor[(1, 10), float32] */;
  add(%669, meta[relay.Constant][125] /* ty=Tensor[(10), float32] */) /* ty=Tensor[(1, 10), float32] */
}
"""

network_wkls = []

conv_2d_count = 0
mp_count = 0

channels_re = re.compile(r"channels=(\d+)")
kernel_re = re.compile(r"kernel_size=\[(\d+), (?:\d+).*")
padding_re = re.compile(r"padding=\[(\d+), (?:\d+), (?:\d+), (?:\d+).*")
stride_re = re.compile(r"strides=\[(\d+), (?:\d+).*")
input_shape_re = re.compile(r"input0: Tensor\[\((?:\d+), (\d+), (\d+), (\d+)\).*")
tensor_shape_re = re.compile(r"Tensor\[\((?:\d+), (\d+), (\d+), (\d+)\)")
ifm_shape_dict = {"height": 224, "width": 224, "channels": 3}
kernel_shape_dict = {"height": 0, "width": 0, "channels": 0, "stride": 0, "padding": 0}

output_file = "dataset/uart_sniffer/asp_dac/demo/resnet18_1x16x16_obf.txt"

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
    # match = input_shape_re.search(line)
    # if match:
    #     input_shape = tuple(map(int, match.groups()))
    #     ifm_shape_dict["height"] = input_shape[1]
    #     ifm_shape_dict["width"] = input_shape[2]
    #     ifm_shape_dict["channels"] = input_shape[0]
    #     continue

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

        # match all the tensor shapes in the line
        tensor_shapes = tensor_shape_re.findall(line)
        input_channels = int(tensor_shapes[0][0])
        oheight, owidth= int(tensor_shapes[1][1]), int(tensor_shapes[1][1])

        # get input shape from conv kernel shape and output shape
        ifm_shape_dict["height"], ifm_shape_dict["width"] = calc_conv_input_size(oheight, owidth,
                                                                                 kernel_shape_dict["height"],
                                                                                 kernel_shape_dict["width"],
                                                                                 kernel_shape_dict["stride"],
                                                                                 kernel_shape_dict["stride"],
                                                                                 kernel_shape_dict["padding"],
                                                                                 kernel_shape_dict["padding"])
        ifm_shape_dict["channels"] = input_channels

        conv_wkl = Conv2DWorkload(batch=env.BATCH, height=ifm_shape_dict["height"], width=ifm_shape_dict["width"], in_filter=ifm_shape_dict["channels"],
                                  out_filter=kernel_shape_dict["channels"], hkernel=kernel_shape_dict["height"],
                                  wkernel=kernel_shape_dict["width"], hstride=kernel_shape_dict["stride"], wstride=kernel_shape_dict["stride"],
                                  hpad=kernel_shape_dict["padding"], wpad=kernel_shape_dict["padding"])


        # conv_out_height, conv_out_width, _ = calc_conv_output_size(conv_wkl)
        # ifm_shape_dict["height"] = conv_out_height
        # ifm_shape_dict["width"] = conv_out_width
        # ifm_shape_dict["channels"] = conv_wkl.out_filter

        if conv_2d_count >= 2:
            network_wkls.append(conv_wkl)

        continue

# print(network_wkls)
generate_labels_file(network_wkls, output_file)



