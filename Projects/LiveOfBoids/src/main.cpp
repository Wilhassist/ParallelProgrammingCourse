#include "../include/utils/MathUtils.hpp"
#include "../include/utils/Vec2.hpp"

#include "../include/agents/AgentFactory.hpp"
#include "../include/config.hpp"

#include "Grid.hpp"
#include <chrono>

int main(int argc, char* argv[]) {
    // Load default configuration
    SimulationConfig config;

    // Parse command-line arguments
    config = parseArguments(argc, argv, config);

    // to be tested resolution
    AgentFactory agentFactory(config);

    agentFactory.createBirds();
    agentFactory.createEagles();
    agentFactory.createPreys();
    
    // global loop
    Grid grid(agentFactory.ratio() * agentFactory.width(), agentFactory.ratio() * agentFactory.height(), config.cellSize, config.margin, config.mode);

    auto allAgents = agentFactory.getAllAgents();
    for (const auto& agent : allAgents) {
        grid.addBoid(agent.get()); 
    }

    auto last_time = std::chrono::steady_clock::now();
    grid.updateBoids();
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(now - last_time).count();

    std::cout<<"NumBoids: "<< config.numAgents + config.numPredators + config.numPreys<<std::endl;
    std::cout<<"Time: "<< duration<<std::endl;

    exit(EXIT_SUCCESS);
}