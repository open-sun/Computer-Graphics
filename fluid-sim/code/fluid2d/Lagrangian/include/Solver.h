#pragma once
#ifndef __LAGRANGIAN_2D_SOLVER_H__
#define __LAGRANGIAN_2D_SOLVER_H__

#include "ParticleSystem2d.h"
#include "Configure.h"

namespace FluidSimulation
{

    namespace Lagrangian2d
    {
        class Solver
        {
        public:
            Solver(ParticleSystem2d &ps);

            void solve();
            void computeDensity();

            // Step 2: Compute pressure
            void computePressure();

            // Step 3: Compute acceleration
            void computeAcceleration();

            // Step 4: Update velocity and position
            void updateVelocityAndPosition();

            // Step 5: Check boundary conditions
            void checkBoundary();

            // Step 6: Update block IDs
            void updateBlockInfo();
            float W(float dist);

        private:
            ParticleSystem2d &mPs;
        };
    }
}

#endif
