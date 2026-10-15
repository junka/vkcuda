// vkpipe_tap.c - LD_PRELOAD shim: dump the exact create-info structs the vendor
// SC sample passes to vkCreateComputePipelines / vkCreatePipelineLayout /
// vkCreateDescriptorSetLayout / vkCreatePipelineCache, then forward to the real
// driver. This gives ground truth for what the 581 SC driver accepts.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

typedef void *VkDevice, *VkPipelineCache, *VkPipeline, *VkPipelineLayout, *VkDescriptorSetLayout;
typedef int32_t VkResult;
typedef uint32_t VkStructureType;

typedef struct { VkStructureType sType; const void *pNext; } Base;

typedef struct {
  VkStructureType sType; const void *pNext; uint32_t flags; uint32_t stage;
  void *module; const char *pName; const uint32_t *pSpec; const void *pSpecData;
} Stage;

typedef struct {
  VkStructureType sType; const void *pNext; uint32_t flags;
  Stage stage; VkPipelineLayout layout; VkPipeline base; int32_t baseIndex;
} CPCI;

typedef struct { uint32_t binding; uint32_t descType; uint32_t descCount; uint32_t stageFlags; const void *pImm; } Binding;

typedef struct {
  VkStructureType sType; const void *pNext; uint32_t flags;
  uint32_t bindingCount; const Binding *pBindings;
} DSLCI;

typedef struct { uint32_t stageFlags; uint32_t offset; uint32_t size; } PCR;

typedef struct {
  VkStructureType sType; const void *pNext; uint32_t flags;
  uint32_t setLayoutCount; const VkDescriptorSetLayout *pSetLayouts;
  uint32_t pushRangeCount; const PCR *pPushRanges;
} PLCI;

typedef struct {
  VkStructureType sType; const void *pNext; uint32_t flags;
  uint64_t dataSize; const void *pData;
} PCCI;

typedef struct {
  VkStructureType sType; const void *pNext; uint8_t id[16];
  uint32_t matchControl; uint64_t poolEntrySize;
} POCI;

static void dump_pnext(const void *p) {
  int depth = 0;
  fprintf(stderr, "[TAP]   pNext chain:\n");
  while (p && depth < 8) {
    const Base *b = (const Base *)p;
    fprintf(stderr, "[TAP]     sType=%d (0x%x)\n", b->sType, b->sType);
    if (b->sType == 1000298010) {
      const POCI *o = (const POCI *)p;
      fprintf(stderr, "[TAP]       OFFLINE id=");
      for (int i = 0; i < 16; i++) fprintf(stderr, "%02x", o->id[i]);
      fprintf(stderr, " matchControl=%u poolEntrySize=%llu\n", o->matchControl,
              (unsigned long long)o->poolEntrySize);
    }
    p = b->pNext; depth++;
  }
}

static void *(*real_gdpa)(VkDevice, const char *);
static VkResult (*real_cpPipe)(VkDevice, VkPipelineCache, uint32_t, const void *, const void *, VkPipeline *);
static VkResult (*real_cpLayout)(VkDevice, const void *, const void *, VkPipelineLayout *);
static VkResult (*real_cpDSL)(VkDevice, const void *, const void *, VkDescriptorSetLayout *);
static VkResult (*real_cpCache)(VkDevice, const void *, const void *, VkPipelineCache *);

typedef VkResult (*PF_cpPipe)(VkDevice, VkPipelineCache, uint32_t, const void *, const void *, VkPipeline *);
typedef VkResult (*PF_cpLayout)(VkDevice, const void *, const void *, VkPipelineLayout *);
typedef VkResult (*PF_cpDSL)(VkDevice, const void *, const void *, VkDescriptorSetLayout *);
typedef VkResult (*PF_cpCache)(VkDevice, const void *, const void *, VkPipelineCache *);
static PF_cpPipe exp_cpPipe; static PF_cpLayout exp_cpLayout; static PF_cpDSL exp_cpDSL; static PF_cpCache exp_cpCache;

static VkResult tap_cpPipe(VkDevice d, VkPipelineCache c, uint32_t n, const void *ci, const void *a, VkPipeline *out) {
  fprintf(stderr, "[TAP] vkCreateComputePipelines n=%u\n", n);
  for (uint32_t i = 0; i < n; i++) {
    const CPCI *k = (const CPCI *)((const char *)ci + i * sizeof(CPCI));
    { const unsigned char *raw=(const unsigned char*)k;
      fprintf(stderr,"[TAP]  raw ci[%u] (%zu bytes):", i, sizeof(CPCI));
      for (size_t z=0; z<sizeof(CPCI); z++) fprintf(stderr,"%s%02x", (z%4)?"":" ", raw[z]);
      fprintf(stderr,"\n"); }
    fprintf(stderr,"[TAP]   stage flags=0x%x pNext=%p pSpecData=%p\n",
            k->stage.flags, k->stage.pNext, k->stage.pSpecData);
    fprintf(stderr, "[TAP]  ci[%u] sType=%d flags=0x%x layout=%p base=%p baseIndex=%d\n",
            i, k->sType, k->flags, (void *)k->layout, (void *)k->base, k->baseIndex);
    fprintf(stderr, "[TAP]   stage sType=%d stage=0x%x module=%p pName=%s pSpec=%p\n",
            k->stage.sType, k->stage.stage, k->stage.module,
            k->stage.pName ? k->stage.pName : "(null)", (void *)k->stage.pSpec);
    dump_pnext(k->pNext);
  }
  VkResult r = exp_cpPipe ? exp_cpPipe(d, c, n, ci, a, out) : real_cpPipe(d, c, n, ci, a, out);
  fprintf(stderr, "[TAP] vkCreateComputePipelines -> %d\n", r);
  return r;
}

static VkResult tap_cpLayout(VkDevice d, const void *ci, const void *a, VkPipelineLayout *out) {
  const PLCI *p = (const PLCI *)ci;
  fprintf(stderr, "[TAP] vkCreatePipelineLayout flags=0x%x setLayoutCount=%u pushRangeCount=%u\n",
          p->flags, p->setLayoutCount, p->pushRangeCount);
  for (uint32_t i = 0; i < p->setLayoutCount && p->pSetLayouts; i++)
    fprintf(stderr, "[TAP]   setLayout[%u]=%p\n", i, (void *)p->pSetLayouts[i]);
  for (uint32_t i = 0; i < p->pushRangeCount && p->pPushRanges; i++)
    fprintf(stderr, "[TAP]   pushRange[%u] stageFlags=0x%x offset=%u size=%u\n", i,
            p->pPushRanges[i].stageFlags, p->pPushRanges[i].offset, p->pPushRanges[i].size);
  dump_pnext(p->pNext);
  VkResult r = exp_cpLayout ? exp_cpLayout(d, ci, a, out) : real_cpLayout(d, ci, a, out);
  fprintf(stderr, "[TAP] vkCreatePipelineLayout -> %d pl=%p\n", r, (void *)*out);
  return r;
}

static VkResult tap_cpDSL(VkDevice d, const void *ci, const void *a, VkDescriptorSetLayout *out) {
  const DSLCI *p = (const DSLCI *)ci;
  fprintf(stderr, "[TAP] vkCreateDescriptorSetLayout flags=0x%x bindingCount=%u\n", p->flags, p->bindingCount);
  for (uint32_t i = 0; i < p->bindingCount && p->pBindings; i++)
    fprintf(stderr, "[TAP]   binding[%u] b=%u type=%u count=%u stageFlags=0x%x\n", i,
            p->pBindings[i].binding, p->pBindings[i].descType, p->pBindings[i].descCount,
            p->pBindings[i].stageFlags);
  dump_pnext(p->pNext);
  VkResult r = exp_cpDSL ? exp_cpDSL(d, ci, a, out) : real_cpDSL(d, ci, a, out);
  fprintf(stderr, "[TAP] vkCreateDescriptorSetLayout -> %d\n", r);
  return r;
}

static VkResult tap_cpCache(VkDevice d, const void *ci, const void *a, VkPipelineCache *out) {
  const PCCI *p = (const PCCI *)ci;
  fprintf(stderr, "[TAP] vkCreatePipelineCache flags=0x%x dataSize=%llu pData=%p\n",
          p->flags, (unsigned long long)p->dataSize, (void *)p->pData);
  VkResult r = exp_cpCache ? exp_cpCache(d, ci, a, out) : real_cpCache(d, ci, a, out);
  fprintf(stderr, "[TAP] vkCreatePipelineCache -> %d pc=%p\n", r, (void *)*out);
  return r;
}

void *vkGetDeviceProcAddr(VkDevice dev, const char *name) {
  if (!real_gdpa) real_gdpa = (void *(*)(VkDevice, const char *))dlsym(RTLD_NEXT, "vkGetDeviceProcAddr");
  if (!strcmp(name, "vkCreateComputePipelines")) {
    if (!real_cpPipe) real_cpPipe = (VkResult (*)(VkDevice, VkPipelineCache, uint32_t, const void *, const void *, VkPipeline *))real_gdpa(dev, name);
    return (void *)tap_cpPipe;
  }
  if (!strcmp(name, "vkCreatePipelineLayout")) {
    if (!real_cpLayout) real_cpLayout = (VkResult (*)(VkDevice, const void *, const void *, VkPipelineLayout *))real_gdpa(dev, name);
    return (void *)tap_cpLayout;
  }
  if (!strcmp(name, "vkCreateDescriptorSetLayout")) {
    if (!real_cpDSL) real_cpDSL = (VkResult (*)(VkDevice, const void *, const void *, VkDescriptorSetLayout *))real_gdpa(dev, name);
    return (void *)tap_cpDSL;
  }
  if (!strcmp(name, "vkCreatePipelineCache")) {
    if (!real_cpCache) real_cpCache = (VkResult (*)(VkDevice, const void *, const void *, VkPipelineCache *))real_gdpa(dev, name);
    return (void *)tap_cpCache;
  }
  return real_gdpa(dev, name);
}


// ---- vkCreateDevice: dump the SC object-reservation create info ----
typedef struct { VkStructureType sType; const void *pNext; uint32_t flags;
  uint32_t queueCount; const void *pQueues; uint32_t layerCount; const void *ppLayers;
  uint32_t extCount; const void *ppExts; const void *pEnabledFeatures; } DCI;

typedef struct {
  VkStructureType sType; const void *pNext;
  uint32_t pipelineCacheCreateInfoCount; const void *pPipelineCacheCreateInfos;
  uint32_t pipelinePoolSizeCount; const void *pPipelinePoolSizes;
  uint32_t *counts; // not used
} RES_HEAD;

typedef struct { VkStructureType sType; const void *pNext; uint64_t poolEntrySize; uint32_t poolEntryCount; } POOLSZ;

typedef struct {
  VkStructureType sType; const void *pNext;
  uint32_t pipelineCacheCreateInfoCount; const void *pPipelineCacheCreateInfos;
  uint32_t pipelinePoolSizeCount; const void *pPipelinePoolSizes;
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
   maxTimestampQueriesPerPool,maxImmutableSamplersPerDescriptorSetLayout;
} RES;

static void dump_res(const void *p) {
  int depth=0;
  static const char *names[] = {
    "semaphoreRequestCount","commandBufferRequestCount","fenceRequestCount",
    "deviceMemoryRequestCount","bufferRequestCount","imageRequestCount","eventRequestCount",
    "queryPoolRequestCount","bufferViewRequestCount","imageViewRequestCount",
    "layeredImageViewRequestCount","pipelineCacheRequestCount","pipelineLayoutRequestCount",
    "renderPassRequestCount","graphicsPipelineRequestCount","computePipelineRequestCount",
    "descriptorSetLayoutRequestCount","samplerRequestCount","descriptorPoolRequestCount",
    "descriptorSetRequestCount","framebufferRequestCount","commandPoolRequestCount",
    "samplerYcbcrConversionRequestCount","surfaceRequestCount","swapchainRequestCount",
    "displayModeRequestCount","subpassDescriptionRequestCount","attachmentDescriptionRequestCount",
    "descriptorSetLayoutBindingRequestCount","descriptorSetLayoutBindingLimit",
    "maxImageViewMipLevels","maxImageViewArrayLayers","maxLayeredImageViewMipLevels",
    "maxOcclusionQueriesPerPool","maxPipelineStatisticsQueriesPerPool",
    "maxTimestampQueriesPerPool","maxImmutableSamplersPerDescriptorSetLayout" };
  while (p && depth<8) {
    const Base *b=(const Base*)p;
    fprintf(stderr,"[TAP]   devpNext sType=%d\n", b->sType);
    if (b->sType == 1000298002) {
      const RES *r=(const RES*)p;
      fprintf(stderr,"[TAP]   RESERVATION cacheCreateInfoCount=%u pCCI=%p poolSizeCount=%u pPool=%p\n",
              r->pipelineCacheCreateInfoCount,(void*)r->pPipelineCacheCreateInfos,
              r->pipelinePoolSizeCount,(void*)r->pPipelinePoolSizes);
      const POOLSZ *ps=(const POOLSZ*)r->pPipelinePoolSizes;
      for (uint32_t i=0;i<r->pipelinePoolSizeCount&&ps;i++)
        fprintf(stderr,"[TAP]     poolSize[%u] sType=%d entrySize=%llu entryCount=%u\n",i,
                ps[i].sType,(unsigned long long)ps[i].poolEntrySize,ps[i].poolEntryCount);
      const uint32_t *c = &r->semaphoreRequestCount;
      for (unsigned i=0;i<sizeof(names)/sizeof(names[0]);i++)
        fprintf(stderr,"[TAP]     %s=%u\n", names[i], c[i]);
    }
    p = b->pNext; depth++;
  }
}

static VkResult (*real_cpDev)(void*, const void*, const void*, VkDevice*);
VkResult vkCreateDevice(void *pd, const void *ci, const void *a, VkDevice *out) {
  if (!real_cpDev) real_cpDev = (VkResult(*)(void*,const void*,const void*,VkDevice*))dlsym(RTLD_NEXT,"vkCreateDevice");
  const DCI *d = (const DCI*)ci;
  fprintf(stderr,"[TAP] vkCreateDevice flags=0x%x queueCount=%u layerCount=%u extCount=%u\n",
          d->flags,d->queueCount,d->layerCount,d->extCount);
  dump_res(d->pNext);
  VkResult r = real_cpDev(pd,ci,a,out);
  fprintf(stderr,"[TAP] vkCreateDevice -> %d dev=%p\n", r, (void*)*out);
  return r;
}

// Exported wrappers for the entry points the sample links directly.
typedef VkResult (*PF_cpCache)(VkDevice, const void *, const void *, VkPipelineCache *);
static PF_cpCache exp_cpCache;
VkResult vkCreatePipelineCache(VkDevice d, const void *ci, const void *a, VkPipelineCache *out) {
  if (!exp_cpCache) exp_cpCache=(PF_cpCache)dlsym(RTLD_NEXT,"vkCreatePipelineCache");
  const PCCI *p=(const PCCI*)ci;
  fprintf(stderr,"[TAP] vkCreatePipelineCache flags=0x%x dataSize=%llu pData=%p\n",
          p->flags,(unsigned long long)p->dataSize,(void*)p->pData);
  VkResult r=exp_cpCache(d,ci,a,out);
  fprintf(stderr,"[TAP] vkCreatePipelineCache -> %d pc=%p\n", r,(void*)*out);
  return r;
}

typedef VkResult (*PF_cpDSL)(VkDevice, const void *, const void *, VkDescriptorSetLayout *);
typedef VkResult (*PF_cpLayout)(VkDevice, const void *, const void *, VkPipelineLayout *);
static PF_cpDSL exp_cpDSL; static PF_cpLayout exp_cpLayout;

VkResult vkCreateDescriptorSetLayout(VkDevice d, const void *ci, const void *a, VkDescriptorSetLayout *out) {
  if (!exp_cpDSL) exp_cpDSL=(PF_cpDSL)dlsym(RTLD_NEXT,"vkCreateDescriptorSetLayout");
  const DSLCI *p=(const DSLCI*)ci;
  fprintf(stderr,"[TAP] vkCreateDescriptorSetLayout flags=0x%x bindingCount=%u\n",p->flags,p->bindingCount);
  for (uint32_t i=0;i<p->bindingCount&&p->pBindings;i++)
    fprintf(stderr,"[TAP]   binding[%u] b=%u type=%u count=%u stageFlags=0x%x\n",i,
            p->pBindings[i].binding,p->pBindings[i].descType,p->pBindings[i].descCount,p->pBindings[i].stageFlags);
  dump_pnext(p->pNext);
  VkResult r=exp_cpDSL(d,ci,a,out);
  fprintf(stderr,"[TAP] vkCreateDescriptorSetLayout -> %d setl=%p\n",r,(void*)*out);
  return r;
}

VkResult vkCreatePipelineLayout(VkDevice d, const void *ci, const void *a, VkPipelineLayout *out) {
  if (!exp_cpLayout) exp_cpLayout=(PF_cpLayout)dlsym(RTLD_NEXT,"vkCreatePipelineLayout");
  const PLCI *p=(const PLCI*)ci;
  fprintf(stderr,"[TAP] vkCreatePipelineLayout flags=0x%x setLayoutCount=%u pushRangeCount=%u\n",
          p->flags,p->setLayoutCount,p->pushRangeCount);
  for (uint32_t i=0;i<p->pushRangeCount&&p->pPushRanges;i++)
    fprintf(stderr,"[TAP]   pushRange[%u] stageFlags=0x%x offset=%u size=%u\n",i,
            p->pPushRanges[i].stageFlags,p->pPushRanges[i].offset,p->pPushRanges[i].size);
  dump_pnext(p->pNext);
  VkResult r=exp_cpLayout(d,ci,a,out);
  fprintf(stderr,"[TAP] vkCreatePipelineLayout -> %d pl=%p\n",r,(void*)*out);
  return r;
}

typedef VkResult (*PF_cpPipe)(VkDevice, VkPipelineCache, uint32_t, const void *, const void *, VkPipeline *);
static PF_cpPipe exp_cpPipe;
VkResult vkCreateComputePipelines(VkDevice d, VkPipelineCache c, uint32_t n, const void *ci, const void *a, VkPipeline *out) {
  if (!exp_cpPipe) exp_cpPipe=(PF_cpPipe)dlsym(RTLD_NEXT,"vkCreateComputePipelines");
  tap_cpPipe(d,c,n,ci,a,out);
  return exp_cpPipe(d,c,n,ci,a,out);
}
