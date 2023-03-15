#include <iostream>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <sstream>
#include <vector>
#include "common.h"
#include "cuda_buffer.h"
#include "LauchParams.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern "C" const unsigned char embedded_ptx_code[];


static void context_log_cb(unsigned int level,
                           const char *tag,
                           const char *message,
                           void *) {
    fprintf(stderr, "[%2d][%12s]: %s\n", (int) level, tag, message);
}
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
};

/*! SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
};

/*! SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    int objectID;
};

int main() {
    cudaFree(0);
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    OPTIX_CHECK(optixInit());

    CUstream stream;
    cudaDeviceProp deviceProps;
    CUcontext cuda_context;


    const int device_id = 0;
    CUDA_CHECK(SetDevice(device_id));
    CUDA_CHECK(StreamCreate(&stream));

    cudaGetDeviceProperties(&deviceProps, device_id);
    std::cout << " running on " << deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&cuda_context);
    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr, "Error querying current context: error code %d\n", cuRes);


    OptixDeviceContext optixContext;
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                        (optixContext, context_log_cb, nullptr, 4));

    std::cout << "create context" << std::endl;

    OptixModule module;
    OptixModuleCompileOptions moduleCompileOptions = {};

    OptixPipeline pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};

    moduleCompileOptions.maxRegisterCount = 50;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    pipelineLinkOptions.maxTraceDepth = 2;

    const std::string ptxCode = reinterpret_cast<const char *>(embedded_ptx_code);

    {
        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
                                             &moduleCompileOptions,
                                             &pipelineCompileOptions,
                                             ptxCode.c_str(),
                                             ptxCode.size(),
                                             log, &sizeof_log,
                                             &module
        ));
        if (sizeof_log > 1) PRINT(log);
        std::cout << "create module" << std::endl;
    }

    std::vector<OptixProgramGroup> raygenPGs;
    CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    {
        raygenPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = module;
        pgDesc.raygen.entryFunctionName = "__raygen__renderFrame";

        char log[2048];
        size_t sizeof_log = sizeof( log );
        // OptixProgramGroup raypg;
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log, &sizeof_log,
                                            &raygenPGs[0]
        ));
        if (sizeof_log > 1) PRINT(log);

    }
    std::cout << "raygen program" << std::endl;

    {
        missPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc    = {};
        pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.miss.module            = module;
        pgDesc.miss.entryFunctionName = "__miss__radiance";

        // OptixProgramGroup raypg;
        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &missPGs[0]
        ));
        if (sizeof_log > 1) PRINT(log);
    }

    {
        hitgroupPGs.resize(1);

        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc    = {};
        pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH            = module;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        pgDesc.hitgroup.moduleAH            = module;
        pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &hitgroupPGs[0]
        ));
        if (sizeof_log > 1) PRINT(log);

        {
            std::vector<OptixProgramGroup> programGroups;
            for (auto pg : raygenPGs)
                programGroups.push_back(pg);
            for (auto pg : missPGs)
                programGroups.push_back(pg);
            for (auto pg : hitgroupPGs)
                programGroups.push_back(pg);

            char log[2048];
            size_t sizeof_log = sizeof( log );
            OPTIX_CHECK(optixPipelineCreate(optixContext,
                                            &pipelineCompileOptions,
                                            &pipelineLinkOptions,
                                            programGroups.data(),
                                            (int)programGroups.size(),
                                            log,&sizeof_log,
                                            &pipeline
            ));
            if (sizeof_log > 1) PRINT(log);

            OPTIX_CHECK(optixPipelineSetStackSize
                                (/* [in] The pipeline to configure the stack size for */
                                        pipeline,
                                        /* [in] The direct stack size requirement for direct
                                           callables invoked from IS or AH. */
                                        2*1024,
                                        /* [in] The direct stack size requirement for direct
                                           callables invoked from RG, MS, or CH.  */
                                        2*1024,
                                        /* [in] The continuation stack requirement. */
                                        2*1024,
                                        /* [in] The maximum depth of a traversable graph
                                           passed to trace. */
                                        1));
            if (sizeof_log > 1) PRINT(log);
        }
    }
    {
        std::vector<RaygenRecord> raygenRecords;
        for (int i=0;i<raygenPGs.size();i++) {
            RaygenRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGs[i],&rec));
            rec.data = nullptr; /* for now ... */
            raygenRecords.push_back(rec);
        }
        raygenRecordsBuffer.alloc_and_upload(raygenRecords);
        sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

        // ------------------------------------------------------------------
        // build miss records
        // ------------------------------------------------------------------
        std::vector<MissRecord> missRecords;
        for (int i=0;i<missPGs.size();i++) {
            MissRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(missPGs[i],&rec));
            rec.data = nullptr; /* for now ... */
            missRecords.push_back(rec);
        }
        missRecordsBuffer.alloc_and_upload(missRecords);
        sbt.missRecordBase          = missRecordsBuffer.d_pointer();
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount         = (int)missRecords.size();

        // ------------------------------------------------------------------
        // build hitgroup records
        // ------------------------------------------------------------------

        // we don't actually have any objects in this example, but let's
        // create a dummy one so the SBT doesn't have any null pointers
        // (which the sanity checks in compilation would complain about)
        int numObjects = 1;
        std::vector<HitgroupRecord> hitgroupRecords;
        for (int i=0;i<numObjects;i++) {
            int objectType = 0;
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPGs[objectType],&rec));
            rec.objectID = i;
            hitgroupRecords.push_back(rec);
        }
        hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt.hitgroupRecordCount         = (int)hitgroupRecords.size();


    }

    LaunchParams launchParams;
    CUDABuffer   launchParamsBuffer;

    launchParamsBuffer.alloc(sizeof(launchParams));

    vec2i newSize{1200,1024};

    CUDABuffer colorBuffer;

    colorBuffer.resize(newSize.x*newSize.y*sizeof(uint32_t));

    // update the launch parameters that we'll pass to the optix
    // launch:
    launchParams.fbSize      = newSize;
    launchParams.colorBuffer = (uint32_t*)colorBuffer.d_ptr;

    launchParamsBuffer.upload(&launchParams,1);
    launchParams.frameID++;

    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
            pipeline,stream,
            /*! parameters and SBT */
            launchParamsBuffer.d_pointer(),
            launchParamsBuffer.sizeInBytes,
            &sbt,
            /*! dimensions of the launch: */
            launchParams.fbSize.x,
            launchParams.fbSize.y,
            1
    ));
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();

    std::vector<uint32_t> pixels(newSize.x*newSize.y);
    colorBuffer.download(pixels.data(),
                         launchParams.fbSize.x*launchParams.fbSize.y);

    const std::string fileName = "osc_example2.png";
    stbi_write_png(fileName.c_str(),newSize.x,newSize.y,4,
                   pixels.data(),newSize.x*sizeof(uint32_t));
    std::cout
              << std::endl
              << "Image rendered, and saved to " << fileName << " ... done." << std::endl
              << std::endl;


}
