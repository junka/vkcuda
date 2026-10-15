// scpipe3.c - like scpipe2 but faithfully reproduces the pipeline create info
// declared in the pcc JSON: real descriptor set layout + pipeline layout
// (incl. push-constant ranges). In Vulkan SC the offline create path still
// validates the app-supplied create info against the cached entry, so an
// empty layout is rejected with VK_ERROR_VALIDATION_FAILED before the UUID
// is ever looked up.
// Args: <cache.bin> <32-hex-uuid> <mode: vadd|vendor> [driver.so]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>
typedef void *VkInstance,*VkPhysicalDevice,*VkDevice,*VkPipeline,*VkPipelineLayout,*VkPipelineCache,*VkShaderModule,*VkDescriptorSetLayout;
typedef uint32_t VkBool32,VkFlags; typedef int32_t VkResult,VkStructureType;
#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO 2
#define VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO 3
#define VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO 18
#define VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO 29
#define VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO 30
#define VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO 17
#define VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO 32
#define VK_STRUCTURE_TYPE_PIPELINE_POOL_SIZE 1000298005
#define VK_STRUCTURE_TYPE_PIPELINE_OFFLINE_CREATE_INFO 1000298010
#define VK_STRUCTURE_TYPE_DEVICE_OBJECT_RESERVATION_CREATE_INFO 1000298002
typedef struct{VkStructureType sType;const void*pNext;}B;
typedef struct{VkStructureType sType;const void*pNext;uint32_t apiVersion;const char*pApplicationName;uint32_t applicationVersion;const char*pEngineName;uint32_t engineVersion;}VkApplicationInfo;
typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;uint32_t qfi,qc;const float*pq;}VkDQCI;
typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;uint32_t cc;const VkDQCI*pq;uint32_t lc;const char*const*pl;uint32_t ec;const char*const*pe;const void*pf;}VkDCI;
typedef struct{VkStructureType sType;const void*pNext;unsigned long long sz;uint32_t cnt;}VkPool;
typedef struct{VkStructureType sType;const void*pNext;
 uint32_t pipelineCacheCreateInfoCount;const void*pPipelineCacheCreateInfos;
 uint32_t pipelinePoolSizeCount;const void*pPipelinePoolSizes;
 uint32_t semaphoreRequestCount,commandBufferRequestCount,fenceRequestCount,
  deviceMemoryRequestCount,bufferRequestCount,imageRequestCount,eventRequestCount,
  queryPoolRequestCount,bufferViewRequestCount,imageViewRequestCount,
  layeredImageViewRequestCount,pipelineCacheRequestCount,pipelineLayoutRequestCount,
  renderPassRequestCount,graphicsPipelineRequestCount,computePipelineRequestCount,
  descriptorSetLayoutRequestCount,samplerRequestCount,descriptorPoolRequestCount,
  descriptorSetRequestCount,framebufferRequestCount,commandPoolRequestCount,
  samplerYcbcrConversionRequestCount,surfaceRequestCount,swapchainRequestCount,
  displayModeRequestCount,subpassDescriptionRequestCount,attachmentDescriptionRequestCount,
  descriptorSetLayoutBindingRequestCount,descriptorSetLayoutBindingLimit,
  maxImageViewMipLevels,maxImageViewArrayLayers,maxLayeredImageViewMipLevels,
  maxOcclusionQueriesPerPool,maxPipelineStatisticsQueriesPerPool,
  maxTimestampQueriesPerPool,maxImmutableSamplersPerDescriptorSetLayout;}VkRes;
typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;uint32_t stage;VkShaderModule mod;const char*pName;const void*pSpec;}VkStage;
typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;VkStage stage;VkPipelineLayout layout;VkPipeline base;int32_t basePipelineIndex;}VkCPCI;
typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;uint32_t setLayoutCount;const VkDescriptorSetLayout*pSetLayouts;uint32_t pushRangeCount;const void*pPushRanges;}VkPLCI;
typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;uint32_t bindingCount;const void*pBindings;}VkDSLCI;
typedef struct{uint32_t binding;VkFlags descriptorType;uint32_t descriptorCount;uint32_t stageFlags;const void*pImmutableSamplers;}VkDSLB;
typedef struct{VkFlags stageFlags;uint32_t offset,size;}VkPCR; // no sType/pNext in VkPushConstantRange
typedef struct{VkStructureType sType;const void*pNext;uint8_t id[16];uint32_t matchControl;unsigned long long poolEntrySize;}VkPOCI;
typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;unsigned long long isz;const void*pdata;}PCCI;
typedef VkResult(*F1)(uint32_t*);typedef VkResult(*F2)(const void*,const void*,VkInstance*);
typedef VkResult(*F3)(VkInstance,uint32_t*,VkPhysicalDevice*);typedef VkResult(*F4)(VkPhysicalDevice,const void*,const void*,VkDevice*);
typedef void*(*F5)(VkDevice,const char*);

static int unhex(const char*s,uint8_t*out){for(int i=0;i<16;i++){unsigned v;if(sscanf(s+2*i,"%2x",&v)!=1)return -1;out[i]=(uint8_t)v;}return 0;}

int main(int c,char**v){setvbuf(stdout,NULL,_IONBF,0);
 if(c<4){printf("usage: %s <cache.bin> <uuid-hex32> <vadd|vendor> [driver.so]\n",v[0]);return 2;}
 uint8_t uuid[16]; if(unhex(v[2],uuid)){printf("bad uuid\n");return 2;}
 int mode_vadd = strcmp(v[3],"vadd")==0;
 FILE*f=fopen(v[1],"rb"); if(!f){printf("open %s failed\n",v[1]);return 1;}
 fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET);
 unsigned char*buf=malloc(sz); if(fread(buf,1,sz,f)!=(size_t)sz){printf("read failed\n");return 1;} fclose(f);
 printf("cache %s = %ld bytes  mode=%s\n",v[1],sz,mode_vadd?"vadd":"vendor");

 const char*dp = c>4?v[4]:"/usr/lib/libnvidia-vksc-core.so.1";
 void*lib=dlopen(dp,RTLD_NOW|RTLD_LOCAL);
 if(!lib){printf("dlopen: %s\n",dlerror());return 1;}
 uint32_t iv=0;((F1)dlsym(lib,"vkEnumerateInstanceVersion"))(&iv);
 VkApplicationInfo app={VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="p";app.apiVersion=iv;
 typedef struct{VkStructureType sType;const void*pNext;VkFlags flags;const VkApplicationInfo*pai;uint32_t lc;const char*const*pl;uint32_t ec;const char*const*pe;}IC;
 IC ic={VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ic.pai=&app;VkInstance inst=0;
 printf("createInstance=%d\n",((F2)dlsym(lib,"vkCreateInstance"))(&ic,0,&inst));
 uint32_t nd=1;VkPhysicalDevice pd;((F3)dlsym(lib,"vkEnumeratePhysicalDevices"))(inst,&nd,&pd);
 float pr=1;VkDQCI q={VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};q.qfi=0;q.qc=1;q.pq=&pr;
 PCCI pcci={VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
 pcci.flags=0x6; pcci.isz=(unsigned long long)sz; pcci.pdata=buf;
 unsigned long long poolsz=1<<20;
 VkPool pool[2]={{VK_STRUCTURE_TYPE_PIPELINE_POOL_SIZE,0,poolsz,1},
                 {VK_STRUCTURE_TYPE_PIPELINE_POOL_SIZE,0,poolsz,1}};
 VkRes res;memset(&res,0,sizeof res);res.sType=VK_STRUCTURE_TYPE_DEVICE_OBJECT_RESERVATION_CREATE_INFO;
 res.pipelinePoolSizeCount=2;res.pPipelinePoolSizes=pool;
 res.pipelineCacheCreateInfoCount=1;res.pPipelineCacheCreateInfos=&pcci;
 res.pipelineCacheRequestCount=2;
 res.pipelineLayoutRequestCount=2;res.computePipelineRequestCount=1;
 res.descriptorSetLayoutRequestCount=2;res.maxImmutableSamplersPerDescriptorSetLayout=0;
 res.descriptorSetLayoutBindingRequestCount=4;
 res.descriptorSetLayoutBindingLimit=mode_vadd?3:2;res.maxImageViewMipLevels=1;
 res.bufferRequestCount=5;res.imageRequestCount=6;res.imageViewRequestCount=5;
 res.semaphoreRequestCount=2;res.commandBufferRequestCount=10;res.fenceRequestCount=4;
 res.deviceMemoryRequestCount=12;res.renderPassRequestCount=1;
 res.graphicsPipelineRequestCount=1;res.samplerRequestCount=2;
 res.descriptorPoolRequestCount=1;res.descriptorSetRequestCount=2;
 res.framebufferRequestCount=2;res.commandPoolRequestCount=2;
 VkDCI d={VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};d.cc=1;d.pq=&q;d.pNext=&res;VkDevice dev=0;
 VkResult r=((F4)dlsym(lib,"vkCreateDevice"))(pd,&d,0,&dev);printf("createDevice=%d\n",r);if(r)return 1;
 F5 gdpa=(F5)dlsym(lib,"vkGetDeviceProcAddr");
 VkResult(*cpLayout)(VkDevice,const void*,const void*,VkPipelineLayout*)=(void*)gdpa(dev,"vkCreatePipelineLayout");
 VkResult(*cpDSL)(VkDevice,const void*,const void*,VkDescriptorSetLayout*)=(void*)gdpa(dev,"vkCreateDescriptorSetLayout");
 VkResult(*cpPipe)(VkDevice,VkPipelineCache,uint32_t,const void*,const void*,VkPipeline*)=(void*)gdpa(dev,"vkCreateComputePipelines");
 VkResult(*cpCache)(VkDevice,const void*,const void*,VkPipelineCache*)=(void*)gdpa(dev,"vkCreatePipelineCache");

 VkPipelineCache pc=0;r=cpCache(dev,&pcci,0,&pc);
 printf("createPipelineCache(from pcc data)=%d pc=%p\n",r,(void*)pc);
 if(r){PCCI e={VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};r=cpCache(dev,&e,0,&pc);
   printf("createPipelineCache(empty)=%d\n",r);}

 // Descriptor set layout exactly as declared in the JSON.
 VkDSLB b[3]; memset(b,0,sizeof b);

 b[0].binding=0; b[0].descriptorCount=1; b[0].descriptorType=7; b[0].stageFlags=0x20; // STORAGE_BUFFER, COMPUTE

 b[1].binding=1; b[1].descriptorCount=1; b[1].descriptorType=mode_vadd?7:6; b[1].stageFlags=0x20; // vadd:SSBO, vendor:UNIFORM_BUFFER

 b[2].binding=2; b[2].descriptorCount=1; b[2].descriptorType=7; b[2].stageFlags=0x20;
 VkDSLCI dsl={VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
 dsl.bindingCount = mode_vadd?3:2; dsl.pBindings=b;
 VkDescriptorSetLayout setl=0; r=cpDSL(dev,&dsl,0,&setl);
 printf("createDescriptorSetLayout(bindings=%u)=%d setl=%p\n",dsl.bindingCount,r,(void*)setl);

 VkPCR pcr;
 pcr.stageFlags=0x20; pcr.offset=0; pcr.size=4;
 VkPLCI plc={VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
 plc.setLayoutCount=1; plc.pSetLayouts=&setl;
 if(mode_vadd){ plc.pushRangeCount=1; plc.pPushRanges=&pcr; }
 else plc.pushRangeCount=0;
 VkPipelineLayout pl=0;
 r=cpLayout(dev,&plc,0,&pl);printf("createPipelineLayout(pc=%u)=%d pl=%p\n",plc.pushRangeCount,r,(void*)pl);

 VkStage st={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};st.stage=0x20;st.pName="main";
 VkPOCI off={VK_STRUCTURE_TYPE_PIPELINE_OFFLINE_CREATE_INFO};off.matchControl=0;off.poolEntrySize=1<<20;
 memcpy(off.id,uuid,16);
 VkCPCI cp={VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage=st;cp.layout=pl;cp.basePipelineIndex=0;cp.pNext=&off;
 VkPipeline pipe=0;r=cpPipe(dev,pc,1,&cp,0,&pipe);
 printf("createComputePipelines(offline, cache UUID)=%d pipe=%p\n",r,(void*)pipe);
 printf("RESULT: %s\n", (r==0&&pipe)?"PIPELINE CREATED FROM pcc CACHE":"FAILED");
 printf("DONE\n");return 0;}
