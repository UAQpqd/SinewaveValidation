//
// Created by dev on 4/9/2018.
//

#include "ProposedMethodology.hpp"

void writeCsvFile(const std::vector<float> *data, unsigned int sps, std::string filename) {
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out);
    ofs << "time,amplitude" << std::endl;
    for (size_t p = 0; p < data->size(); p++) {
        float time = (float) p / (float) sps;
        float amplitude = data->at(p);
        ofs << time << "," << amplitude << std::endl;
    }
    ofs.close();
}

void DEBlock::run(unsigned int sps,
                  bc::context &ctx) {
    // Copy signal to device
    bc::vector<float> dSignal(inSignal->begin(),inSignal->end(),*queue);
    const float sumOfSquares =
            std::accumulate(inSignal->begin(),inSignal->end(), 0.0f,
                            [](float accum, float val) -> float {
                                accum = accum + val * val;
                            });
    // Set signal as parameter
    kernel->set_arg(0,dSignal);
    kernel->set_arg(4,(unsigned int)dSignal.size());
    kernel->set_arg(5,sumOfSquares);
    kernel->set_arg(6,inOmegaMin);
    kernel->set_arg(7,inOmegaMax);
    // Initialize population
    auto dX = bc::vector<bc::float2_>(config.popSize, ctx);
    auto dScores = bc::vector<float>(config.popSize, ctx);
    kernel->set_arg(1,dX);
    kernel->set_arg(2,dScores);
    kernel->set_arg(3,(unsigned int)dX.size());
    // Estimate A by the sumOfSquares
    /*float omega = (inOmegaMax+inOmegaMin)/2.0f;
    float T = (float)dSignal.size()/(float)sps;
    double delta = 1.0/(double)sps;
    double integrationApprox =
            std::accumulate(inSignal->begin(),inSignal->end(), 0.0,
                            [&delta](double accum, double val) -> double {
                                accum = accum + pow(delta*val,2.0);
                            });
    float a = sqrt((float)integrationApprox*2.0f/T);*/

    kernel->set_arg(10,inA); //TODO: Check
    auto evaluatePopulationLambda =
            [this,/*&queue, &kernel, */&dSignal, &dScores, &dX, &sumOfSquares](
                    MinusDarwin::Scores &scores,
                    const MinusDarwin::Population &population) {
                //Population to Device
                std::vector<bc::float2_> hPopulation(population.size());
                std::transform(population.begin(),population.end(),
                               hPopulation.begin(),[](const MinusDarwin::Agent &a){
                            bc::float2_ b;
                            b[0] = a.at(0);
                            b[1] = a.at(1);
                            return b;
                        });
                bc::copy(hPopulation.begin(),hPopulation.end(),dX.begin(),*queue);
                //Once population has been copied to the device
                //a parallel calculation of scores must be done
                queue->enqueue_1d_range_kernel(*kernel,0,hPopulation.size(),0);
                bc::copy(dScores.begin(),dScores.end(),scores.begin(),*queue);
            };
    MinusDarwin::Solver solver(config,evaluatePopulationLambda);
    auto agent = solver.run(false);
    outEstOmega = inOmegaMin+(inOmegaMax-inOmegaMin)*agent.at(0);
    outEstPhi = 2.0f*M_PI*agent.at(1);
    std::cout << "Frequency:" << outEstOmega/(2.0f*M_PI) << " "
              << "Phi:" << outEstPhi/(2.0f*M_PI) << std::endl;
}

void RMSBlock::run(unsigned int sps) {
    // For each half cycle RMS is calculated
    double T = (double)inSignal->size()/(double)sps;
    double freq = (double)inOmega/(2.0*M_PI);
    double delta = 1.0/(freq*2.0);
    double hcCountDbl = T*freq;
    unsigned int hcCount = (unsigned int)floor(hcCountDbl);
    for(unsigned int hc = 0; hc<hcCount; hc++)
    {
        unsigned int hcStart =
                floor(delta*(double)hc*(double)inSignal->size()/T);
        unsigned int hcEnd =
                floor(delta*(double)(hc+1)*(double)inSignal->size()/T);

        if(hcEnd>=inSignal->size()) break;
        float sum = std::accumulate(
                inSignal->begin()+(int)hcStart,
                inSignal->begin()+(int)hcEnd,
                0.0f,
                [](float accum, float val) -> float {
                    accum = accum + pow(val,2.0f);
                });
        float N = (float)(hcEnd-hcStart);
        outRMS.push_back(sqrt(sum/N));
    }
}

ProposedMethodology::ProposedMethodology(
        const unsigned int t_sps,
        const float t_epsilon,
        const float t_freq0star,
        const float t_freqSearchAbsTol,
        const std::vector<float> t_harmonicFactors) :
        sps(t_sps), epsilon(t_epsilon),
        freq0star(t_freq0star),
        freqSearchAbsTol(t_freqSearchAbsTol) {
    // Get all filters initialized
    fp1.setup(flickerFilterOrder,sps,freq0star-freqSearchAbsTol/*,1*/);
    fp2.setup(fundamentalFilterOrder,sps,freq0star,2.0f*freqSearchAbsTol/*,1*/);
    harmonicFactors = std::vector<float>(
            t_harmonicFactors.begin(),
            t_harmonicFactors.end());
    for(auto &factor : harmonicFactors) {

    }
    // Initialize RMS blocks
    //harmonicsRMSBlocks.resize(harmonicFactors.size());
}

void ProposedMethodology::run(const std::vector<float> *rp) {
    // Get GPU device ready for compute
    auto dev = bc::system::default_device();
    auto ctx = bc::context(dev);
    auto queue = bc::command_queue(ctx,dev);
    auto program = bc::program::create_with_source(
            sinewaveFitSourceParallel,ctx);
    try {
        program.build();
    } catch(std::exception e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }
    auto kernel = bc::kernel(
            program,"calculateScoresOfPopulation");
    kernel.set_arg(8,2.0f*(float)M_PI);
    kernel.set_arg(9,(unsigned int)sps);
    writeCsvFile(rp,sps,"rp.csv");
    // Filter rp by the low pass filter
    std::vector<float> rp0(rp->begin(),rp->end());
    float *channels[1];
    channels[0] = &rp0.at(0);
    fp1.process((int)rp0.size(),channels);
    writeCsvFile(&rp0,sps,"rp0.csv");
    std::vector<float> rp1(rp->begin(),rp->end());
    channels[0] = &rp1.at(0);
    fp2.process((int)rp1.size(),channels);
    writeCsvFile(&rp1,sps,"rp1.csv");
    // Search from 0.05Hz to fundamental frequency lower bound
    flickerDEBlock = new DEBlock(
            &rp0,
            0.05f,
            (freq0star-freqSearchAbsTol)*2.0f*(float)M_PI,
            1.0f,
            &queue,
            &kernel);
    std::cout << "Flicker ";
    flickerDEBlock->run(sps, ctx);
    // Search on the fundamental frequency range
    fundamentalDEBlock = new DEBlock(
            &rp1,
            (freq0star-freqSearchAbsTol)*2.0f*(float)M_PI,
            (freq0star+freqSearchAbsTol)*2.0f*(float)M_PI,
            127.0f*sqrt(2.0f),
            &queue,
            &kernel);
    std::cout << "Fundamental ";
    fundamentalDEBlock->run(sps, ctx);
    /* Once flicker and fundamental parameters are had
     * the RMS of each half cycle of them must be
     * calculated as for harmonic frequencies too */
    flickerRmsBlock = new RMSBlock(
            &rp0,
            flickerDEBlock->outEstOmega
    );
    flickerRmsBlock->run(sps);
    float flickerMeanRms = std::accumulate(
            flickerRmsBlock->outRMS.begin(),
            flickerRmsBlock->outRMS.end(),
            0.0f,std::plus<float>()
    );
    flickerMeanRms /= (float)flickerRmsBlock->outRMS.size();
    std::cout << "Flicker mean RMS: " << flickerMeanRms << std::endl;
    fundamentalRmsBlock = new RMSBlock(
            rp,
            fundamentalDEBlock->outEstOmega
    );
    fundamentalRmsBlock->run(sps);


    float fundamentalMeanRms = std::accumulate(
            fundamentalRmsBlock->outRMS.begin(),
            fundamentalRmsBlock->outRMS.end(),
            0.0f,std::plus<float>()
    );
    fundamentalMeanRms /= (float)fundamentalRmsBlock->outRMS.size();
    std::cout << "Fundamental mean RMS: " << fundamentalMeanRms << std::endl;

    for(auto factor : harmonicFactors) {
        std::vector<float> harmonicSignal(rp->begin(),rp->end());
        Dsp::SimpleFilter<Dsp::Butterworth::BandPass<harmonicFilterOrder>,1> filter;
        filter.setup(
                harmonicFilterOrder,sps,
                factor*fundamentalDEBlock->outEstOmega/(2.0f*M_PI),
                2.0f*freqSearchAbsTol/*,1*/);
        channels[0] = &harmonicSignal.at(0);
        filter.process((int)harmonicSignal.size(),channels);
        auto harmonicRmsBlock = new RMSBlock(
                &harmonicSignal,
                fundamentalDEBlock->outEstOmega
        );
        harmonicRmsBlock->run(sps);

        float harmonicMeanRms = std::accumulate(
                harmonicRmsBlock->outRMS.begin(),
                harmonicRmsBlock->outRMS.end(),
                0.0f,std::plus<float>()
        );
        harmonicMeanRms /= (float)harmonicRmsBlock->outRMS.size();
        std::cout << "Harmonic " << factor << " mean RMS: " << harmonicMeanRms << std::endl;

    }
}
