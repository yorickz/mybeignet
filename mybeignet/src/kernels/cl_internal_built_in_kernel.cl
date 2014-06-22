kernel void __cl_copy_region_align4 ( global float* src, unsigned int src_offset,
                                     global float* dst, unsigned int dst_offset,
				     unsigned int size)
{
    int i = get_global_id(0);
    if (i < size)
        dst[i+dst_offset] = src[i+src_offset];
}
kernel void __cl_copy_region_align16 ( global float* src, unsigned int src_offset,
                                      global float* dst, unsigned int dst_offset,
				      unsigned int size)
{
    int i = get_global_id(0) * 4;
    if (i < size*4) {
        dst[i+dst_offset] = src[i+src_offset];
        dst[i+dst_offset + 1] = src[i+src_offset + 1];
        dst[i+dst_offset + 2] = src[i+src_offset + 2];
        dst[i+dst_offset + 3] = src[i+src_offset + 3];
    }
}
kernel void __cl_copy_region_unalign_same_offset ( global int* src, unsigned int src_offset,
                                     global int* dst, unsigned int dst_offset,
				     unsigned int size,
				     unsigned int first_mask, unsigned int last_mask)
{
    int i = get_global_id(0);
    if (i > size -1)
       return;

    if (i == 0) {
        dst[dst_offset] = (dst[dst_offset] & first_mask)
             | (src[src_offset] & (~first_mask));
    } else if (i == size - 1) {
        dst[i+dst_offset] = (src[i+src_offset] & last_mask)
            | (dst[i+dst_offset] & (~last_mask));
    } else {
        dst[i+dst_offset] = src[i+src_offset];
    }
}
kernel void __cl_copy_region_unalign_dst_offset ( global int* src, unsigned int src_offset,
                                     global int* dst, unsigned int dst_offset,
				     unsigned int size,
				     unsigned int first_mask, unsigned int last_mask,
				     unsigned int shift, unsigned int dw_mask)
{
    int i = get_global_id(0);
    unsigned int tmp = 0;

    if (i > size -1)
        return;

    /* last dw, need to be careful, not to overflow the source. */
    if ((i == size - 1) && ((last_mask & (~(~dw_mask >> shift))) == 0)) {
        tmp = ((src[src_offset + i] & ~dw_mask) >> shift);
    } else {
        tmp = ((src[src_offset + i] & ~dw_mask) >> shift)
             | ((src[src_offset + i + 1] & dw_mask) << (32 - shift));
    }

    if (i == 0) {
        dst[dst_offset] = (dst[dst_offset] & first_mask) | (tmp & (~first_mask));
    } else if (i == size - 1) {
        dst[i+dst_offset] = (tmp & last_mask) | (dst[i+dst_offset] & (~last_mask));
    } else {
        dst[i+dst_offset] = tmp;
    }
}
kernel void __cl_copy_region_unalign_src_offset ( global int* src, unsigned int src_offset,
                                     global int* dst, unsigned int dst_offset,
				     unsigned int size,
				     unsigned int first_mask, unsigned int last_mask,
				     unsigned int shift, unsigned int dw_mask, int src_less)
{
    int i = get_global_id(0);
    unsigned int tmp = 0;

    if (i > size -1)
        return;

    if (i == 0) {
        tmp = ((src[src_offset + i] & dw_mask) << shift);
    } else if (src_less && i == size - 1) { // not exceed the bound of source
        tmp = ((src[src_offset + i - 1] & ~dw_mask) >> (32 - shift));
    } else {
        tmp = ((src[src_offset + i - 1] & ~dw_mask) >> (32 - shift))
             | ((src[src_offset + i] & dw_mask) << shift);
    }

    if (i == 0) {
        dst[dst_offset] = (dst[dst_offset] & first_mask) | (tmp & (~first_mask));
    } else if (i == size - 1) {
        dst[i+dst_offset] = (tmp & last_mask) | (dst[i+dst_offset] & (~last_mask));
    } else {
        dst[i+dst_offset] = tmp;
    }
}
kernel void __cl_copy_buffer_rect ( global char* src, global char* dst,
                                          unsigned int region0, unsigned int region1, unsigned int region2,
                                          unsigned int src_offset, unsigned int dst_offset,
                                          unsigned int src_row_pitch, unsigned int src_slice_pitch,
                                          unsigned int dst_row_pitch, unsigned int dst_slice_pitch)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  src_offset += k * src_slice_pitch + j * src_row_pitch + i;
  dst_offset += k * dst_slice_pitch + j * dst_row_pitch + i;
  dst[dst_offset] = src[src_offset];
}
kernel void __cl_copy_image_2d_to_2d(__read_only image2d_t src_image, __write_only image2d_t dst_image,
                             unsigned int region0, unsigned int region1, unsigned int region2,
                             unsigned int src_origin0, unsigned int src_origin1, unsigned int src_origin2,
                             unsigned int dst_origin0, unsigned int dst_origin1, unsigned int dst_origin2)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  int4 color;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
  int2 src_coord;
  int2 dst_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  src_coord.x = src_origin0 + i;
  src_coord.y = src_origin1 + j;
  dst_coord.x = dst_origin0 + i;
  dst_coord.y = dst_origin1 + j;
  color = read_imagei(src_image, sampler, src_coord);
  write_imagei(dst_image, dst_coord, color);
}
kernel void __cl_copy_image_3d_to_2d(__read_only image3d_t src_image, __write_only image2d_t dst_image,
                             unsigned int region0, unsigned int region1, unsigned int region2,
                             unsigned int src_origin0, unsigned int src_origin1, unsigned int src_origin2,
                             unsigned int dst_origin0, unsigned int dst_origin1, unsigned int dst_origin2)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  int4 color;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
  int4 src_coord;
  int2 dst_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  src_coord.x = src_origin0 + i;
  src_coord.y = src_origin1 + j;
  src_coord.z = src_origin2 + k;
  dst_coord.x = dst_origin0 + i;
  dst_coord.y = dst_origin1 + j;
  color = read_imagei(src_image, sampler, src_coord);
  write_imagei(dst_image, dst_coord, color);
}
kernel void __cl_copy_image_2d_to_3d(__read_only image2d_t src_image, __write_only image3d_t dst_image,
                                         unsigned int region0, unsigned int region1, unsigned int region2,
                                         unsigned int src_origin0, unsigned int src_origin1, unsigned int src_origin2,
                                         unsigned int dst_origin0, unsigned int dst_origin1, unsigned int dst_origin2)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  int4 color;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
  int2 src_coord;
  int4 dst_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  src_coord.x = src_origin0 + i;
  src_coord.y = src_origin1 + j;
  dst_coord.x = dst_origin0 + i;
  dst_coord.y = dst_origin1 + j;
  dst_coord.z = dst_origin2 + k;
  color = read_imagei(src_image, sampler, src_coord);
  write_imagei(dst_image, dst_coord, color);
}
kernel void __cl_copy_image_3d_to_3d(__read_only image3d_t src_image, __write_only image3d_t dst_image,
                             unsigned int region0, unsigned int region1, unsigned int region2,
                             unsigned int src_origin0, unsigned int src_origin1, unsigned int src_origin2,
                             unsigned int dst_origin0, unsigned int dst_origin1, unsigned int dst_origin2)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  int4 color;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
  int4 src_coord;
  int4 dst_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  src_coord.x = src_origin0 + i;
  src_coord.y = src_origin1 + j;
  src_coord.z = src_origin2 + k;
  dst_coord.x = dst_origin0 + i;
  dst_coord.y = dst_origin1 + j;
  dst_coord.z = dst_origin2 + k;
  color = read_imagei(src_image, sampler, src_coord);
  write_imagei(dst_image, dst_coord, color);
}
kernel void __cl_copy_image_2d_to_buffer( __read_only image2d_t image, global uchar* buffer,
                                        unsigned int region0, unsigned int region1, unsigned int region2,
                                        unsigned int src_origin0, unsigned int src_origin1, unsigned int src_origin2,
                                        unsigned int dst_offset)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  uint4 color;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
  int2 src_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  src_coord.x = src_origin0 + i;
  src_coord.y = src_origin1 + j;
  color = read_imageui(image, sampler, src_coord);
  dst_offset += (k * region1 + j) * region0 + i;
  buffer[dst_offset] = color.x;
}
#define IMAGE_TYPE image3d_t
#define COORD_TYPE int4
kernel void __cl_copy_image_3d_to_buffer ( __read_only IMAGE_TYPE image, global uchar* buffer,
                                        unsigned int region0, unsigned int region1, unsigned int region2,
                                        unsigned int src_origin0, unsigned int src_origin1, unsigned int src_origin2,
                                        unsigned int dst_offset)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  uint4 color;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
  COORD_TYPE src_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  src_coord.x = src_origin0 + i;
  src_coord.y = src_origin1 + j;
  src_coord.z = src_origin2 + k;
  color = read_imageui(image, sampler, src_coord);
  dst_offset += (k * region1 + j) * region0 + i;
  buffer[dst_offset] = color.x;
}
kernel void __cl_copy_buffer_to_image_2d(__read_only image2d_t image, global uchar* buffer,
                                        unsigned int region0, unsigned int region1, unsigned int region2,
                                        unsigned int dst_origin0, unsigned int dst_origin1, unsigned int dst_origin2,
                                        unsigned int src_offset)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  uint4 color = (uint4)(0);
  int2 dst_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  dst_coord.x = dst_origin0 + i;
  dst_coord.y = dst_origin1 + j;
  src_offset += (k * region1 + j) * region0 + i;
  color.x = buffer[src_offset];
  write_imageui(image, dst_coord, color);
}
kernel void __cl_copy_buffer_to_image_3d(__read_only image3d_t image, global uchar* buffer,
                                        unsigned int region0, unsigned int region1, unsigned int region2,
                                        unsigned int dst_origin0, unsigned int dst_origin1, unsigned int dst_origin2,
                                        unsigned int src_offset)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  uint4 color = (uint4)(0);
  int4 dst_coord;
  if((i >= region0) || (j>= region1) || (k>=region2))
    return;
  dst_coord.x = dst_origin0 + i;
  dst_coord.y = dst_origin1 + j;
  dst_coord.z = dst_origin2 + k;
  src_offset += (k * region1 + j) * region0 + i;
  color.x = buffer[src_offset];
  write_imageui(image, dst_coord, color);
}
