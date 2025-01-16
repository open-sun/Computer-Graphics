#include "fluid3d/Eulerian/include/Solver.h"
#include "Configure.h"
#include "Global.h"

namespace FluidSimulation
{
    namespace Eulerian3d
    {
        Solver::Solver(MACGrid3d& grid) : mGrid(grid)
        {
            mGrid.reset();  // Reset grid to initial state
        }

        void Solver::solve()
        {

        }

    }
}