#include <spconv/spconv_ops.h>
namespace spconv {

std::vector<torch::Tensor>
getIndicePairs(torch::Tensor indices, int64_t batchSize,
               std::vector<int64_t> outSpatialShape,
               std::vector<int64_t> spatialShape,
               std::vector<int64_t> kernelSize, std::vector<int64_t> stride,
               std::vector<int64_t> padding, std::vector<int64_t> dilation,
               std::vector<int64_t> outPadding, int64_t _subM,
               int64_t _transpose, int64_t _useHash) {
  // auto timer = spconv::CudaContextTimer<>();
  bool subM = _subM != 0;
  bool transpose = _transpose != 0;
  auto NDim = kernelSize.size();
  // CPU always use hash (tsl::robin_map).
  bool useHash = _useHash != 0 || indices.device().type() == torch::kCPU;
  auto numAct = indices.size(0);
  auto coorDim = indices.size(1) - 1; // batchIdx + xyz
  TV_ASSERT_RT_ERR(NDim == coorDim, "error");
  TV_ASSERT_RT_ERR(kernelSize.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outSpatialShape.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(stride.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(padding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(outPadding.size() == coorDim, "error");
  TV_ASSERT_RT_ERR(dilation.size() == coorDim, "error");
  auto kernelVolume = kernelSize[0];
  for (int i = 1; i < kernelSize.size(); ++i) {
    kernelVolume *= kernelSize[i];
  }
  TV_ASSERT_RT_ERR(kernelVolume <= 4096, "error");
  auto outputVolume = outSpatialShape[0];
  for (int i = 1; i < outSpatialShape.size(); ++i) {
    outputVolume *= outSpatialShape[i];
  }
  std::string msg = "due to limits of cuda hash, the volume of dense space "
                    "include batch size ";
  msg += "must less than std::numeric_limits<int>::max() = 2e9";
  TV_ASSERT_RT_ERR(batchSize * outputVolume < std::numeric_limits<int>::max(),
                   msg);
  torch::Tensor indicePairs =
      torch::full({2, kernelVolume, numAct}, -1,
                  torch::dtype(torch::kInt32).device(indices.device()));
  torch::Tensor indiceNum = torch::zeros(
      {kernelVolume}, torch::dtype(torch::kInt32).device(indices.device()));
  auto gridSize = batchSize * outputVolume;
  if (useHash) {
    gridSize = batchSize;
  }
  torch::Tensor gridOut = torch::full(
      {gridSize}, -1, torch::dtype(torch::kInt32).device(indices.device()));
  gridOut = gridOut.view({batchSize, -1});
  int64_t numActOut = -1;
  for (int i = 0; i < NDim; ++i) {
    if (subM) {
      padding[i] = kernelSize[i] / 2;
      stride[i] = 1;
    }
  }
  // tv::ssprint("prepare", timer.report() / 1000.0);
  if (subM) {
    if (indices.device().type() == torch::kCPU) {
      numActOut = create_submconv_indice_pair_cpu(
          indices, gridOut, indicePairs, indiceNum, kernelSize, stride, padding,
          dilation, outSpatialShape, transpose, false, useHash);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      numActOut = create_submconv_indice_pair_cuda(
          indices, gridOut, indicePairs, indiceNum, kernelSize, stride, padding,
          dilation, outSpatialShape, transpose, false, useHash);
      if (numActOut == -1) {
        auto device = indices.device();
        indicePairs = indicePairs.to({torch::kCPU});
        indiceNum = indiceNum.to({torch::kCPU});
        indices = indices.to({torch::kCPU});
        numActOut = create_submconv_indice_pair_cpu(
            indices, gridOut, indicePairs, indiceNum, kernelSize, stride,
            padding, dilation, outSpatialShape, transpose, false, useHash);
        return {indices.to(device), indicePairs.to(device),
                indiceNum.to(device)};
      }

    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // tv::ssprint("subm", timer.report() / 1000.0);
    return {indices, indicePairs, indiceNum};
  } else {
    auto indicePairUnique = torch::full(
        {indicePairs.numel() / 2 + 1}, std::numeric_limits<int>::max(),
        torch::dtype(torch::kInt32).device(indices.device()));
    torch::Tensor outInds =
        torch::zeros({numAct * kernelVolume, coorDim + 1},
                     torch::dtype(torch::kInt32).device(indices.device()));
    if (indices.device().type() == torch::kCPU) {
      numActOut = create_conv_indice_pair_cpu(
          indices, outInds, gridOut, indicePairs, indiceNum, kernelSize, stride,
          padding, dilation, outSpatialShape, transpose, false, useHash);
    }
#ifdef TV_CUDA
    else if (indices.device().type() == torch::kCUDA) {
      numActOut = create_conv_indice_pair_p1_cuda(
          indices, indicePairs, indiceNum, indicePairUnique, kernelSize, stride,
          padding, dilation, outSpatialShape, transpose);
      if (numActOut > 0) {
        auto res = torch::_unique(indicePairUnique);
        indicePairUnique = std::get<0>(res);
        numActOut = create_conv_indice_pair_p2_cuda(
            indices, outInds, gridOut, indicePairs, indiceNum, indicePairUnique,
            outSpatialShape, transpose, false, useHash);
        if (numActOut == -1) {
          auto device = indices.device();
          outInds = outInds.to({torch::kCPU});
          indicePairs = indicePairs.to({torch::kCPU});
          indiceNum = indiceNum.to({torch::kCPU});
          indices = indices.to({torch::kCPU});
          numActOut = create_conv_indice_pair_cpu(
              indices, outInds, gridOut, indicePairs, indiceNum, kernelSize,
              stride, padding, dilation, outSpatialShape, transpose, false,
              useHash);

          return {outInds.to(device).slice(0, 0, numActOut),
                  indicePairs.to(device), indiceNum.to(device)};
        }
      }
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    return {outInds.slice(0, 0, numActOut), indicePairs, indiceNum};
  }
}

torch::Tensor indiceConv(torch::Tensor features, torch::Tensor filters,
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM,
                         int64_t algo) {
  auto kernelVolume = indiceNum.size(0);
  switch (algo) {
  case kBatchGemmGather:
  case kBatch: {
    if (kernelVolume != 1) {
      return indiceConvBatch(features, filters, indicePairs, indiceNum,
                             numActOut, _inverse, _subM,
                             algo != kBatchGemmGather);
    } else {
      break;
    }
  }
  case kNative:
    break;
  default:
    TV_THROW_RT_ERR("unknown algo");
  }
  // auto timer = spconv::CudaContextTimer<>();

  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});

  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
  filters = filters.view({-1, numInPlanes, numOutPlanes});

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2;
  int indicePairMaxSize = numActOut;
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    torch::mm_out(output, features, filters[indicePairMaxOffset]);

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      return output;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + kernelVolume);
  }

  torch::Tensor inputBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::empty({indicePairMaxSize, numOutPlanes}, options);

  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  // tv::ssprint("first subm gemm time", timer.report() / 1000.0);

  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    // TODO torch::from_blob is a little slow
    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob =
        torch::from_blob(inputBuffer.data_ptr(), {nHot, numInPlanes}, options);

    if (device == torch::kCPU) {
      sparse_gather_cpu(inputBuffer, features, indicePairs[inverse][i], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
      /* slower than SparseGatherFunctor, may due to int->long conversion
      auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
      auto indicePairBlob = torch::from_blob(indicePairLong.data<long>(),
      {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
      features, 0, indicePairBlob);*/
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // totalGatherTime += timer.report() / 1000.0;
    torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
    // totalGEMMTime += timer.report() / 1000.0;

    if (device == torch::kCPU) {
      sparse_scatter_add_cpu(outputBuffer, output, indicePairs[!inverse][i],
                             nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_scatter_add_cuda(outputBuffer, output, indicePairs[!inverse][i],
                              nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // totalSAddTime += timer.report() / 1000.0;
  }
  // tv::ssprint(totalGatherTime, totalGEMMTime, totalSAddTime);
  return output;
}

torch::Tensor indiceConvBatch(torch::Tensor features, torch::Tensor filters,
                              torch::Tensor indicePairs,
                              torch::Tensor indiceNum, int64_t numActOut,
                              int64_t _inverse, int64_t _subM,
                              bool batchScatter) {
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto kernelVolume = indiceNum.size(0);
  TV_ASSERT_INVALID_ARG(kernelVolume > 1, "error");
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  // auto timer = spconv::CudaContextTimer<>();
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto indicePairNumVec =
      std::vector<int>(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumVec.begin(), indicePairNumVec.end());
  int indicePairMaxOffset = indicePairMaxSizeIter - indicePairNumVec.begin();
  int indicePairMaxSize = *indicePairMaxSizeIter;
  std::nth_element(indicePairNumVec.begin(), indicePairNumVec.begin() + 1,
                   indicePairNumVec.end(), std::greater<int>());
  int indicePairTop2Size = indicePairNumVec[1];
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  auto indice_dtype = indicePairs.scalar_type();
  torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
  // we cant use batch conv in subm directly because
  // number of indice in the center of filter is much more than other
  // filter location.
  // so we first use top2 indice num to do batch conv, then
  // do native conv (gemm) in center.
  int bufferSize = subM ? indicePairTop2Size : indicePairMaxSize;
  int maxKernelVolumePart = kernelVolume;
  std::vector<std::pair<int, int>> part_ranges = {{0, kernelVolume}};
  filters = filters.view({kernelVolume, numInPlanes, numOutPlanes});

  if (subM) {
    maxKernelVolumePart = std::max(indicePairMaxOffset,
                                   int(kernelVolume - indicePairMaxOffset - 1));
    part_ranges = {{0, indicePairMaxOffset},
                   {indicePairMaxOffset + 1, kernelVolume}};
    torch::mm_out(output, features, filters[indicePairMaxOffset]);
    if (indicePairTop2Size == 0) {
      return output;
    }
  }
  // tv::ssprint("first subm gemm time", timer.report() / 1000.0);
  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;

  torch::Tensor inputBuffer =
      torch::empty({maxKernelVolumePart, bufferSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::empty({maxKernelVolumePart, bufferSize, numOutPlanes}, options);
  for (auto &range : part_ranges) {
    int start = range.first;
    int end = range.second;
    int length = end - start;
    int64_t size = length * bufferSize;
    auto inputBufferPart = tv::torch_slice_first_axis(inputBuffer, 0, length);
    auto outputBufferPart = tv::torch_slice_first_axis(outputBuffer, 0, length);
    auto indicePairs1Part =
        tv::torch_slice_first_axis(indicePairs[inverse], start, end);
    auto indicePairs2Part =
        tv::torch_slice_first_axis(indicePairs[!inverse], start, end);
    auto filtersPart = tv::torch_slice_first_axis(filters, start, end);
    if (device == torch::kCPU) {
      TV_THROW_INVALID_ARG("unknown device type");
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      batch_sparse_gather_cuda(inputBufferPart, features, indicePairs1Part,
                               size);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // totalGatherTime += timer.report() / 1000.0;

    torch::bmm_out(outputBufferPart, inputBufferPart, filtersPart);
    // totalGEMMTime += timer.report() / 1000.0;

    if (batchScatter) {
      if (device == torch::kCPU) {
        TV_THROW_INVALID_ARG("unknown device type");
      }
#ifdef TV_CUDA
      else if (device == torch::kCUDA) {
        batch_sparse_scatter_add_cuda(outputBufferPart, output,
                                      indicePairs2Part, size);
      }
#endif
      else {
        TV_THROW_INVALID_ARG("unknown device type");
      }
    } else {
      for (int i = 0; i < length; ++i) {
        auto nHot = indicePairNumCpu.data_ptr<int>()[i + start];
        if (nHot <= 0) {
          continue;
        }
        if (device == torch::kCPU) {
          sparse_scatter_add_cpu(outputBufferPart[i], output,
                                 indicePairs2Part[i], nHot);
        }
#ifdef TV_CUDA
        else if (device == torch::kCUDA) {
          sparse_scatter_add_cuda(outputBufferPart[i], output,
                                  indicePairs2Part[i], nHot);
        }
#endif
        else {
          TV_THROW_INVALID_ARG("unknown device type");
        }
      }
    }
    // totalSAddTime += timer.report() / 1000.0;
  }
  // tv::ssprint(totalGatherTime, totalGEMMTime, totalSAddTime);

  return output;
}

std::vector<torch::Tensor>
indiceConvBackward(torch::Tensor features, torch::Tensor filters,
                   torch::Tensor outGrad, torch::Tensor indicePairs,
                   torch::Tensor indiceNum, int64_t _inverse, int64_t _subM,
                   int64_t algo) {
  auto kernelVolume = indiceNum.size(0);
  switch (algo) {
  case kBatchGemmGather:
  case kBatch: {
    if (kernelVolume != 1) {
      return indiceConvBackwardBatch(features, filters, outGrad, indicePairs,
                                     indiceNum, _inverse, _subM,
                                     algo != kBatchGemmGather);
    } else {
      break;
    }
  }
  case kNative:
    break;
  default:
    TV_THROW_RT_ERR("unknown algo");
  }

  bool subM = _subM != 0;
  bool inverse = _inverse != 0;

  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  auto filterShape = filters.sizes();
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  torch::Tensor filtersGrad = torch::empty(filterShape, options);

  filters = filters.view({-1, numInPlanes, numOutPlanes});
  filtersGrad = filtersGrad.view({-1, numInPlanes, numOutPlanes});

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2;
  int indicePairMaxSize = indicePairNumCpu.data_ptr<int>()[indicePairMaxOffset];
  if (subM) {
    auto filterGradSub = filtersGrad[indicePairMaxOffset];
    torch::mm_out(filterGradSub, features.t(), outGrad);
    torch::mm_out(inputGrad, outGrad, filters[indicePairMaxOffset].t());

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      return {inputGrad, filtersGrad.view(filterShape)};
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + kernelVolume);
  }

  torch::Tensor inputBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::empty({indicePairMaxSize, numOutPlanes}, options);

  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    if (device == torch::kCPU) {
      sparse_gather_cpu(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cpu(outputBuffer, outGrad, indicePairs[!inverse][i], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cuda(outputBuffer, outGrad, indicePairs[!inverse][i], nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }

    auto filterGradSub = filtersGrad[i];
    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob =
        torch::from_blob(inputBuffer.data_ptr(), {nHot, numInPlanes}, options);

    torch::mm_out(filterGradSub, inputBufferBlob.t(), outputBufferBlob);
    torch::mm_out(inputBufferBlob, outputBufferBlob, filters[i].t());
    if (device == torch::kCPU) {
      sparse_scatter_add_cpu(inputBuffer, inputGrad, indicePairs[inverse][i],
                             nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_scatter_add_cuda(inputBuffer, inputGrad, indicePairs[inverse][i],
                              nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
  }
  return {inputGrad, filtersGrad.view(filterShape)};
}

std::vector<torch::Tensor>
indiceConvBackwardBatch(torch::Tensor features, torch::Tensor filters,
                        torch::Tensor outGrad, torch::Tensor indicePairs,
                        torch::Tensor indiceNum, int64_t _inverse,
                        int64_t _subM, bool batchScatter) {
  bool subM = _subM != 0;
  bool inverse = _inverse != 0;

  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto kernelVolume = indiceNum.size(0);
  TV_ASSERT_INVALID_ARG(kernelVolume > 1, "error");
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto indicePairNumVec =
      std::vector<int>(indicePairNumCpu.data_ptr<int>(),
                       indicePairNumCpu.data_ptr<int>() + kernelVolume);
  auto indicePairMaxSizeIter =
      std::max_element(indicePairNumVec.begin(), indicePairNumVec.end());
  int indicePairMaxOffset = indicePairMaxSizeIter - indicePairNumVec.begin();
  int indicePairMaxSize = *indicePairMaxSizeIter;
  std::nth_element(indicePairNumVec.begin(), indicePairNumVec.begin() + 1,
                   indicePairNumVec.end(), std::greater<int>());
  int indicePairTop2Size = indicePairNumVec[1];
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  auto indice_dtype = indicePairs.scalar_type();
  auto filterShape = filters.sizes();
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  torch::Tensor filtersGrad = torch::zeros(filterShape, options);
  int bufferSize = subM ? indicePairTop2Size : indicePairMaxSize;

  filters = filters.view({-1, numInPlanes, numOutPlanes});
  filtersGrad = filtersGrad.view({-1, numInPlanes, numOutPlanes});

  std::vector<std::pair<int, int>> part_ranges = {{0, kernelVolume}};
  int maxKernelVolumePart = kernelVolume;
  if (subM) {
    maxKernelVolumePart = std::max(indicePairMaxOffset,
                                   int(kernelVolume - indicePairMaxOffset - 1));
    part_ranges = {{0, indicePairMaxOffset},
                   {indicePairMaxOffset + 1, kernelVolume}};
    auto filtersGradSub = filtersGrad[indicePairMaxOffset];
    auto filtersSub = filters[indicePairMaxOffset];
    torch::mm_out(filtersGradSub, features.t(), outGrad);
    torch::mm_out(inputGrad, outGrad, filtersSub.t());
    if (indicePairTop2Size == 0) {
      return {inputGrad, filtersGrad.view(filterShape)};
    }
  }
  torch::Tensor inputBuffer =
      torch::zeros({maxKernelVolumePart, bufferSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::zeros({maxKernelVolumePart, bufferSize, numOutPlanes}, options);

  for (auto &range : part_ranges) {
    int start = range.first;
    int end = range.second;
    int length = end - start;
    int64_t size = length * bufferSize;
    auto inputBufferPart = tv::torch_slice_first_axis(inputBuffer, 0, length);
    auto outputBufferPart = tv::torch_slice_first_axis(outputBuffer, 0, length);
    auto indicePairs1Part =
        tv::torch_slice_first_axis(indicePairs[inverse], start, end);
    auto indicePairs2Part =
        tv::torch_slice_first_axis(indicePairs[!inverse], start, end);
    auto filtersPart = tv::torch_slice_first_axis(filters, start, end);
    auto filtersGradPart = tv::torch_slice_first_axis(filtersGrad, start, end);

    if (device == torch::kCPU) {
      TV_THROW_INVALID_ARG("unknown device type");
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      batch_sparse_gather_cuda(inputBufferPart, features, indicePairs1Part,
                               size);
      batch_sparse_gather_cuda(outputBufferPart, outGrad, indicePairs2Part,
                               size);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // filters: KV, I, O, inputBuffer: [KV, buffer, I]
    // outputBuffer: [KV, buffer, O]
    torch::bmm_out(filtersGradPart, inputBufferPart.permute({0, 2, 1}),
                   outputBufferPart);
    torch::bmm_out(inputBuffer, outputBufferPart,
                   filtersPart.permute({0, 2, 1}));
    if (batchScatter) {
      if (device == torch::kCPU) {
        TV_THROW_INVALID_ARG("unknown device type");
      }
#ifdef TV_CUDA
      else if (device == torch::kCUDA) {
        batch_sparse_scatter_add_cuda(inputBufferPart, inputGrad,
                                      indicePairs1Part, size);
      }
#endif
      else {
        TV_THROW_INVALID_ARG("unknown device type");
      }
    } else {
      for (int i = 0; i < length; ++i) {
        auto nHot = indicePairNumCpu.data_ptr<int>()[i + start];
        if (nHot <= 0) {
          continue;
        }
        if (device == torch::kCPU) {
          sparse_scatter_add_cpu(inputBufferPart[i], inputGrad,
                                 indicePairs1Part[i], nHot);
        }
#ifdef TV_CUDA
        else if (device == torch::kCUDA) {
          sparse_scatter_add_cuda(inputBufferPart[i], inputGrad,
                                  indicePairs1Part[i], nHot);
        }
#endif
        else {
          TV_THROW_INVALID_ARG("unknown device type");
        }
      }
    }
  }
  return {inputGrad, filtersGrad.view(filterShape)};
}



torch::Tensor indiceTransInvariantConv(torch::Tensor features, torch::Tensor filters, 
                         torch::Tensor centerFeatures,
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM,
                         int64_t algo) {
  // 只用在输入的第一层.
  auto kernelVolume = indiceNum.size(0);
  switch (algo) {
  case kBatchGemmGather:
  case kBatch: { // not implemented this type
    if (kernelVolume != 1) {
      return indiceConvBatch(features, filters, indicePairs, indiceNum,
                             numActOut, _inverse, _subM,
                             algo != kBatchGemmGather);
    } else {
      break;
    }
  }
  case kNative: // only implemented this type
    break;
  default:
    TV_THROW_RT_ERR("unknown algo");
  }
  // auto timer = spconv::CudaContextTimer<>();

  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto numActIn = features.size(0);

  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
  filters = filters.view({-1, numInPlanes, numOutPlanes});

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2;
  int indicePairMaxSize = numActOut;
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    // torch::mm_out(output, features, filters[indicePairMaxOffset]);
    // 这里numActIn == numActOut
    torch::Tensor tmp_features = torch::zeros({numActIn, numInPlanes}, options);
    // 这里是两个tensor相减
    torch::sub_out(tmp_features, features, centerFeatures);
    // std::cout << "for filter " << indicePairMaxOffset << ", features[0]: " << features[0] << std::endl;
    // std::cout << "for filter " << indicePairMaxOffset << ", centerFeatures[0]: " << centerFeatures[0] << std::endl;
    // std::cout << "for filter " << indicePairMaxOffset << ", tmp_features[0]: " << tmp_features[0] << std::endl;

    torch::mm_out(output, tmp_features, filters[indicePairMaxOffset]);

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      return output;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + kernelVolume);
  }

  torch::Tensor inputBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::empty({indicePairMaxSize, numOutPlanes}, options);
  torch::Tensor centerBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor tmpBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);

  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  // tv::ssprint("first subm gemm time", timer.report() / 1000.0);

  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    // TODO torch::from_blob is a little slow
    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr(), 
                                             {nHot, numInPlanes}, options);
    auto centerBufferBlob = torch::from_blob(centerBuffer.data_ptr(),
                                             {nHot, numInPlanes}, options);
    auto tmpBufferBlob = torch::from_blob(tmpBuffer.data_ptr(),
                                             {nHot, numInPlanes}, options);

    if (device == torch::kCPU) {
      // 这里不能先减再gather, 而是应该先gather出来再减
      sparse_gather_cpu(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cpu(centerBufferBlob, centerFeatures, indicePairs[!inverse][i], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cuda(centerBufferBlob, centerFeatures, indicePairs[!inverse][i], nHot);

      // 同时gather 一个centerBuffer, 然后再相减
      // sparse_gather_cuda(centerBuffer, centerFeatures, indicePairs[!inverse][i], nHot);
      // 要注意这个centerBuffer.size[0]应该要和inputBuffer完全一直.
      // std::cout << "centerBuffer.size(0): " << centerBuffer.size(0) <<", inputBuffer.size(0): " << inputBuffer.size(0) << std::endl;

      /* slower than SparseGatherFunctor, may due to int->long conversion
      auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
      auto indicePairBlob = torch::from_blob(indicePairLong.data<long>(),
      {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
      features, 0, indicePairBlob);*/
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // totalGatherTime += timer.report() / 1000.0;
    // 好像不能加在这里, 会对0也造成影响, 不过要看看inputBufferBlob的shape
    // std::cout << "for filter " << i << ", inputBufferBlob.size(0): " << inputBufferBlob.size(0) <<", centerBufferBlob.size(0):"<< centerBufferBlob.size(0) << std::endl;
    // std::cout << "for filter " << i << ", inputBufferBlob[0]: " << inputBufferBlob[0] << std::endl;
    // std::cout << "for filter " << i << ", centerBufferBlob[0]: " << centerBufferBlob[0] << std::endl;

    // 看看inputBufferBlob的shape, 如果是包含了0的话就不能在这里减;
    // 如果没有包含0的话才可以在这里减;
    // 这里是否应该需要额外定义一个tmpBufferBlob = inputBufferBlob - centerBufferBlob? 
    torch::sub_out(tmpBufferBlob, inputBufferBlob, centerBufferBlob);
    // std::cout << "for filter " << i << ", tmpBufferBlob[0]: " << tmpBufferBlob[0] << std::endl;

    torch::mm_out(outputBufferBlob, tmpBufferBlob, filters[i]);
    // totalGEMMTime += timer.report() / 1000.0;

    if (device == torch::kCPU) {
      sparse_scatter_add_cpu(outputBuffer, output, indicePairs[!inverse][i],
                             nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_scatter_add_cuda(outputBuffer, output, indicePairs[!inverse][i],
                              nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // totalSAddTime += timer.report() / 1000.0;
  }
  // tv::ssprint(totalGatherTime, totalGEMMTime, totalSAddTime);
  return output;
}


std::vector<torch::Tensor>
indiceTransInvariantConvBackward(torch::Tensor features, torch::Tensor filters,
                   torch::Tensor centerFeatures,
                   torch::Tensor outGrad, torch::Tensor indicePairs,
                   torch::Tensor indiceNum, int64_t _inverse, int64_t _subM,
                   int64_t algo) {
  auto kernelVolume = indiceNum.size(0);
  switch (algo) {
  case kBatchGemmGather:
  case kBatch: {
    if (kernelVolume != 1) {
      return indiceConvBackwardBatch(features, filters, outGrad, indicePairs,
                                     indiceNum, _inverse, _subM,
                                     algo != kBatchGemmGather);
    } else {
      break;
    }
  }
  case kNative: // only implemented this type
    break;
  default:
    TV_THROW_RT_ERR("unknown algo");
  }

  bool subM = _subM != 0;
  bool inverse = _inverse != 0;

  auto device = features.device().type();
  auto ndim = filters.dim() - 2;
  auto numInPlanes = features.size(1);
  auto numOutPlanes = filters.size(ndim + 1);
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  auto filterShape = filters.sizes();
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  torch::Tensor filtersGrad = torch::empty(filterShape, options);
  torch::Tensor tmpGrad = torch::zeros(features.sizes(), options);

  filters = filters.view({-1, numInPlanes, numOutPlanes});
  filtersGrad = filtersGrad.view({-1, numInPlanes, numOutPlanes});

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2;
  int indicePairMaxSize = indicePairNumCpu.data_ptr<int>()[indicePairMaxOffset];
  if (subM) {
    auto filterGradSub = filtersGrad[indicePairMaxOffset];
    torch::mm_out(filterGradSub, features.t(), outGrad);
    // torch::mm_out(inputGrad, outGrad, filters[indicePairMaxOffset].t());
    // outGrad -> tmpGrad -> inputGrad and centerGrad
    torch::mm_out(tmpGrad, outGrad, filters[indicePairMaxOffset].t());
    // centerFeatures or centerFeatures.t()?
    torch::sub_out(inputGrad, tmpGrad, centerFeatures);
    // TODO: 是否应该计算这个centerGrad?
    // NOTE: 现在没有计算centerGrad, 当作是一个标量来处理的. 
    // torch::sub_out(centerGrad, tmpGrad, features); 

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      return {inputGrad, filtersGrad.view(filterShape)};
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + kernelVolume);
  }

  torch::Tensor inputBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::empty({indicePairMaxSize, numOutPlanes}, options);
  torch::Tensor centerBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor tmpBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);


  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    if (device == torch::kCPU) {
      sparse_gather_cpu(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cpu(outputBuffer, outGrad, indicePairs[!inverse][i], nHot);
      sparse_gather_cpu(centerBuffer, centerFeatures, indicePairs[!inverse][i], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cuda(outputBuffer, outGrad, indicePairs[!inverse][i], nHot);
      sparse_gather_cuda(centerBuffer, centerFeatures, indicePairs[!inverse][i], nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }

    auto filterGradSub = filtersGrad[i];
    auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
                                             {nHot, numOutPlanes}, options);
    auto inputBufferBlob = torch::from_blob(inputBuffer.data_ptr(), 
                                             {nHot, numInPlanes}, options);
    auto centerBufferBlob = torch::from_blob(centerBuffer.data_ptr(), 
                                             {nHot, numInPlanes}, options);
    auto tmpBufferBlob = torch::from_blob(tmpBuffer.data_ptr(), 
                                             {nHot, numInPlanes}, options);

    torch::mm_out(filterGradSub, inputBufferBlob.t(), outputBufferBlob);
    // torch::mm_out(inputBufferBlob, outputBufferBlob, filters[i].t());
    torch::mm_out(tmpBufferBlob, outputBufferBlob, filters[i].t());
    torch::sub_out(inputBufferBlob, tmpBufferBlob, centerBufferBlob);
    
    
    if (device == torch::kCPU) {
      sparse_scatter_add_cpu(inputBuffer, inputGrad, indicePairs[inverse][i],
                             nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_scatter_add_cuda(inputBuffer, inputGrad, indicePairs[inverse][i],
                              nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
  }
  return {inputGrad, filtersGrad.view(filterShape)};
}



torch::Tensor indiceGroup(torch::Tensor features, 
                         torch::Tensor indicePairs, torch::Tensor indiceNum,
                         int64_t numActOut, int64_t _inverse, int64_t _subM,
                         int64_t algo) {
  auto kernelVolume = indiceNum.size(0);
  switch (algo) {
  case kBatchGemmGather:
  case kBatch: 
  case kNative:
    break;
  default:
    TV_THROW_RT_ERR("unknown algo");
  }
  // auto timer = spconv::CudaContextTimer<>();

  bool subM = _subM != 0;
  bool inverse = _inverse != 0;
  auto device = features.device().type();
  auto ndim = 3; // 暂时只支持3d data
  auto numInPlanes = features.size(1);
  // auto numOutPlanes = filters.size(ndim + 1);
  auto numOutPlanes = features.size(1); // group 操作不改变通道数目
  auto indicePairNumCpu = indiceNum.to({torch::kCPU});

  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());

  // torch::Tensor output = torch::zeros({numActOut, numOutPlanes}, options);
  // filters = filters.view({-1, numInPlanes, numOutPlanes});

  // output：(N_ks, N_b, C)
  torch::Tensor output = torch::zeros({kernelVolume, numActOut, numOutPlanes}, options);
  // torch::Tensor output = torch::zeros({numActOut, numInPlanes}, options);



  // init for subM
  int indicePairMaxOffset = kernelVolume / 2;
  int indicePairMaxSize = numActOut;
  if (subM) { // the center index of subm conv don't need gather and scatter
    // add.
    // torch::mm_out(output, features, filters[indicePairMaxOffset]);
    // 直接赋值给kernel的这个位置？
    output[indicePairMaxOffset] = features;

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      // NOTE: 目前还没处理这个，如果进入这里的话实际上是报错了
      return output;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + kernelVolume);
  }

  torch::Tensor inputBuffer =
      torch::empty({indicePairMaxSize, numInPlanes}, options);
  
  // group 操作暂时不需要这个outputBuffer
  // torch::Tensor outputBuffer =
      // torch::empty({indicePairMaxSize, numOutPlanes}, options);

  double totalGatherTime = 0;
  double totalGEMMTime = 0;
  double totalSAddTime = 0;
  // tv::ssprint("first subm gemm time", timer.report() / 1000.0);


  // gather的作用: 是直接从features上收集到inputBuffer
  // scatter的作用: 是从outputBuffer上放到output上
  // 但对于group来说, inputBuffer与outputBuffer之间没有计算操作, 可以等同起来吗？

  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    // 对于(subM && i == indicePairMaxOffset), 因为之前已经赋值过了, 所以暂时不再处理
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    // TODO torch::from_blob is a little slow
    // 不再需要outputBufferBlob和inputBufferBlob, 因为不再有mm_out
    // auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
                                            //  {nHot, numOutPlanes}, options);
    // auto inputBufferBlob =
        // torch::from_blob(inputBuffer.data_ptr(), {nHot, numInPlanes}, options);

    if (device == torch::kCPU) {
      sparse_gather_cpu(inputBuffer, features, indicePairs[inverse][i], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
      /* slower than SparseGatherFunctor, may due to int->long conversion
      auto indicePairLong = indicePairs[i][inverse].to(torch::kInt64);
      auto indicePairBlob = torch::from_blob(indicePairLong.data<long>(),
      {nHot}, indicePairOptions); torch::index_select_out(inputBufferBlob,
      features, 0, indicePairBlob);*/
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // totalGatherTime += timer.report() / 1000.0;
    // torch::mm_out(outputBufferBlob, inputBufferBlob, filters[i]);
    // totalGEMMTime += timer.report() / 1000.0;

    if (device == torch::kCPU) {
      // scatter_add 也是和初始化的0相加，所以应该是没影响的吧？
      // outputBuffer -> output[i], inputBuffer等同于outputBuffer 
      sparse_scatter_add_cpu(inputBuffer, output[i], indicePairs[!inverse][i],
                             nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      // scatter_add 也是和初始化的0相加，所以应该是没影响的吧？
      // outputBuffer -> output[i], inputBuffer等同于outputBuffer 
      sparse_scatter_add_cuda(inputBuffer, output[i], indicePairs[!inverse][i],
                              nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
    // totalSAddTime += timer.report() / 1000.0;
  }
  // tv::ssprint(totalGatherTime, totalGEMMTime, totalSAddTime);
  return output;
}

torch::Tensor indiceGroupBackward(torch::Tensor features,
                   torch::Tensor outGrad, torch::Tensor indicePairs,
                   torch::Tensor indiceNum, int64_t _inverse, int64_t _subM,
                   int64_t algo) {
  auto kernelVolume = indiceNum.size(0);
  switch (algo) {
  case kBatchGemmGather:
  case kBatch:
  case kNative:
    break;
  default:
    TV_THROW_RT_ERR("unknown algo");
  }

  bool subM = _subM != 0;
  bool inverse = _inverse != 0;

  auto device = features.device().type();

  // auto ndim = filters.dim() - 2;
  auto ndim = 3; // 暂时只支持3d data

  auto numInPlanes = features.size(1);
  // auto numOutPlanes = filters.size(ndim + 1);
  auto numOutPlanes = features.size(1);

  auto indicePairNumCpu = indiceNum.to({torch::kCPU});
  auto options =
      torch::TensorOptions().dtype(features.dtype()).device(features.device());
  // auto filterShape = filters.sizes();
  torch::Tensor inputGrad = torch::zeros(features.sizes(), options);
  // torch::Tensor filtersGrad = torch::empty(filterShape, options);

  // filters = filters.view({-1, numInPlanes, numOutPlanes});
  // filtersGrad = filtersGrad.view({-1, numInPlanes, numOutPlanes});

  // init for subM
  int indicePairMaxOffset = kernelVolume / 2;
  int indicePairMaxSize = indicePairNumCpu.data_ptr<int>()[indicePairMaxOffset];
  if (subM) {
    // auto filterGradSub = filtersGrad[indicePairMaxOffset];
    // torch::mm_out(filterGradSub, features.t(), outGrad);
    // torch::mm_out(inputGrad, outGrad, filters[indicePairMaxOffset].t());
    // 梯度采用从哪来就回到哪去的原则，如果对于同一个点收到来自多个group的梯度，就采用累加的方式
    // 那这里不就只能接收到最后一次的梯度了吗？==> 不会的, 后面会在这个基础上进行累加
    // NOTE: 那么现在的问题是, 梯度究竟应该累加还是均分呢？
    inputGrad = outGrad[indicePairMaxOffset];

    // get indice pair second max size based on subM symmetric property
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + indicePairMaxOffset);
    if (indicePairMaxSize == 0) {
      // 这里也是异常的出口，还没处理。
      return inputGrad;
    }
  } else {
    indicePairMaxSize =
      *std::max_element(indicePairNumCpu.data_ptr<int>(),
                        indicePairNumCpu.data_ptr<int>() + kernelVolume);
  }

  // group 操作暂时不需要这个inputBuffer
  // torch::Tensor inputBuffer =
  //     torch::empty({indicePairMaxSize, numInPlanes}, options);
  torch::Tensor outputBuffer =
      torch::empty({indicePairMaxSize, numOutPlanes}, options);

  for (int i = 0; i < kernelVolume; ++i) {
    auto nHot = indicePairNumCpu.data_ptr<int>()[i];
    if (nHot <= 0 || (subM && i == indicePairMaxOffset)) {
      continue;
    }
    if (device == torch::kCPU) {
      // sparse_gather_cpu(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cpu(outputBuffer, outGrad[i], indicePairs[!inverse][i], nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      // sparse_gather_cuda(inputBuffer, features, indicePairs[inverse][i], nHot);
      sparse_gather_cuda(outputBuffer, outGrad[i], indicePairs[!inverse][i], nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }

    // auto filterGradSub = filtersGrad[i];
    // auto outputBufferBlob = torch::from_blob(outputBuffer.data_ptr(),
                                            //  {nHot, numOutPlanes}, options);
    // auto inputBufferBlob =
        // torch::from_blob(inputBuffer.data_ptr(), {nHot, numInPlanes}, options);

    // torch::mm_out(filterGradSub, inputBufferBlob.t(), outputBufferBlob);
    // torch::mm_out(inputBufferBlob, outputBufferBlob, filters[i].t());
    // NOTE: 这里应该采用scatter_add or scatter_mean 呢？ 
    if (device == torch::kCPU) {
      sparse_scatter_add_cpu(outputBuffer, inputGrad, indicePairs[inverse][i],
                             nHot);
    }
#ifdef TV_CUDA
    else if (device == torch::kCUDA) {
      sparse_scatter_add_cuda(outputBuffer, inputGrad, indicePairs[inverse][i],
                              nHot);
    }
#endif
    else {
      TV_THROW_INVALID_ARG("unknown device type");
    }
  }
  // return {inputGrad}; // for vector output
  return inputGrad;
}



} // namespace spconv
