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
    float omega = (inOmegaMax+inOmegaMin)/2.0f;
    float T = (float)dSignal.size()/(float)sps;
    double delta = 1.0/(double)sps;
    double integrationApprox =
            std::accumulate(inSignal->begin(),inSignal->end(), 0.0,
                            [&delta](double accum, double val) -> double {
                                accum = accum + pow(delta*val,2.0);
                            });
    float a = sqrt((float)integrationApprox*2.0f/T);

    kernel->set_arg(10,a); //TODO: Check
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
    auto agent = solver.run(true);
    outEstOmega = agent.at(0);
    outEstPhi = agent.at(1);
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
    fp1.setup(3,sps,freq0star+freqSearchAbsTol,1);
    fp2.setup(3,sps,freq0star-freqSearchAbsTol,1);
    /*fp1.setup(3,sps,freq0star+freqSearchAbsTol);
    fp2.setup(3,sps,freq0star-freqSearchAbsTol);*/
    harmonicFactors = std::vector<float>(
            t_harmonicFactors.begin(),
            t_harmonicFactors.end());
    for(auto &factor : harmonicFactors) {
        Dsp::SimpleFilter<Dsp::ChebyshevII::BandPass<3>,1> filter;
        filter.setup(3,sps,factor*freq0star,2.0f*freqSearchAbsTol,1);
        harmonicFilters.push_back(filter);
    }
    // Initialize RMS blocks
    harmonicsRMSBlocks.resize(harmonicFactors.size());
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
    std::vector<float> rplow(rp->begin(),rp->end());
    float *channels[1];
    channels[0] = &rplow.at(0);
    fp1.process((int)rplow.size(),channels);
    // Filter rp0 by the high pass filter
    std::vector<float> rp1(rplow.begin(),rplow.end());
    channels[0] = &rp1.at(0);
    fp2.process((int)rp1.size(),channels);
    writeCsvFile(&rp1,sps,"rp1.csv");
    std::vector<float> rp0(rp1.size());
    for(size_t p = 0;p<rp0.size();p++)
        rp0.at(p) = rplow.at(p) - rp1.at(p);
    writeCsvFile(&rp0,sps,"rp0.csv");
    flickerDEBlock = new DEBlock(
            &rp0,
            0.05f,
            (freq0star-freqSearchAbsTol)*2.0f*(float)M_PI,
            &queue,
            &kernel);
    flickerDEBlock->run(sps, ctx);
    fundamentalDEBlock = new DEBlock(
            &rp1,
            (freq0star-freqSearchAbsTol)*2.0f*(float)M_PI,
            (freq0star+freqSearchAbsTol)*2.0f*(float)M_PI,
            &queue,
            &kernel);
    fundamentalDEBlock->run(sps, ctx);
}
