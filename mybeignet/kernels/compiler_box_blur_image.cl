__kernel void compiler_box_blur_image(__read_only image2d_t src,
                                      __write_only image2d_t dst)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                            CLK_ADDRESS_CLAMP_TO_EDGE |
                            CLK_FILTER_NEAREST;
  const int2 coord = (int2)(get_global_id(0), get_global_id(1));
  int2 offset;
  float4 sum = 0;

  for (offset.y = -1; offset.y <= 1; offset.y++) {
    for (offset.x = -1; offset.x <= 1; offset.x++) {
      sum +=  read_imagef(src, sampler, coord + offset);
    }
  }

  write_imagef(dst, coord, (1.0f/9.0f)*sum);
}
