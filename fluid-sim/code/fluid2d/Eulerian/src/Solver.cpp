#include "Eulerian/include/Solver.h"
#include "Configure.h"
#include <iostream>

namespace FluidSimulation
{
    namespace Eulerian2d
    {
        Solver::Solver(MACGrid2d& grid) : mGrid(grid)
        {
            mGrid.reset();  
        }

        void Solver::solve()
        {
            float dt = Eulerian2dPara::dt;  
            Adjection(dt);
            ExternalForces(dt);
            Projection();
            BoundaryConditions();
          
   
        }

        void Solver::Adjection(float dt)
        {
            Glb::CubicGridData2d newd = mGrid.mD;
            Glb::CubicGridData2d newt = mGrid.mT;

            Glb::GridData2dX newU = mGrid.mU;
            Glb::GridData2dY newV = mGrid.mV;
            FOR_EACH_CELL
            {


                   glm::vec2 center = mGrid.getCenter(i, j);
                   glm::vec2 newPos = mGrid.semiLagrangian(center, dt);
                   newd(i, j) = mGrid.mD.interpolate(newPos);
                   newt(i, j) = mGrid.mT.interpolate(newPos);




            }
                FOR_EACH_LINE


            {
              
                    if (mGrid.isValid(i, j, MACGrid2d::X)&&i<mGrid.dim[0])//不加会飘出去
                    {
                             glm::vec2 left = mGrid.getLeft(i, j);
                            glm::vec2 newPos1 = mGrid.semiLagrangian(left, dt);
                            newU(i, j) = mGrid.getVelocityX(newPos1);
                    }
                    if (mGrid.isValid(i, j, MACGrid2d::Y) && j < mGrid.dim[1])
                    {
                        glm::vec2 bottom = mGrid.getBottom(i, j);
                        glm::vec2 newPos = mGrid.semiLagrangian(bottom, dt);
                        newV(i, j) = mGrid.getVelocityY(newPos);

                    }
            
             }

            mGrid.mU = newU;
            mGrid.mT = newt;
            mGrid.mD = newd;
            mGrid.mV = newV;

            
        }

        void Solver::ExternalForces(float dt)
        {

                Glb::GridData2dY newV = mGrid.mV;
                FOR_EACH_LINE
                {
                     if (mGrid.isValid(i, j, MACGrid2d::Y))
                    {
                         double newyadd = mGrid.getBoussinesqForce(mGrid.getCenter(i, j)) * dt;
                             newV(i, j) += newyadd;
                             newV(i, j)-= 9.8*dt;
                     }
                
                }
                
                mGrid.mV = newV;
            }

            void Solver::Projection() {
                int maxdiedai = 100; 
                double upline = 1e-6;

                double dt = Eulerian2dPara::dt; 
              

                std::vector<std::vector<double>> mP(mGrid.dim[0],
                    std::vector<double>(mGrid.dim[1], 0.0));

                for (int zhouqi = 0; zhouqi < maxdiedai; zhouqi++) {
                   
                    double maxChange = 0.0;
                    std::vector<std::vector<double>> newP = mP;

                    FOR_EACH_LINE
                    {

                        if (mGrid.isSolidCell(i, j)) {
                            mGrid.mU(i, j) = 0.0f;
                            mGrid.mV(i, j) = 0.0f;//不加会在边界发生奇怪的东西
                            continue;
                        }

                        double divergence = mGrid.getDivergence(i, j); // 计算散度
                        double presssum = 0.0;
                        int count = 0;//不加会在边界出现问题


                        // 上方邻居
                        if (i - 1 >= 0 && !mGrid.isSolidCell(i - 1, j)) {
                            presssum += mP[i - 1][j];
                            count++;
                        }

                        // 下方邻居
                        if (i + 1 < mGrid.dim[0] && !mGrid.isSolidCell(i + 1, j)) {
                            presssum += mP[i + 1][j];
                            count++;
                            
                        }

                        // 左侧邻居
                        if (j - 1 >= 0 && !mGrid.isSolidCell(i, j - 1)) {
                            presssum += mP[i][j - 1];
                            count++;
                          
                        }

                        // 右侧邻居
                        if (j + 1 < mGrid.dim[1] && !mGrid.isSolidCell(i, j + 1)) {
                            presssum += mP[i][j + 1];
                            count++;
                           
                        }
                      
                            newP[i][j] = (presssum- divergence * Eulerian2dPara::airDensity * mGrid.cellSize * mGrid.cellSize / dt) /count;
                            double change = std::fabs(newP[i][j] - mP[i][j]);
                            maxChange = max(change, maxChange);
                        
                    }
                    mP = newP;

                    if (maxChange < upline) {
                        break;
                    }
                }
                
                
                FOR_EACH_LINE{
                    double changeX = 0.0;
                    double changeY = 0.0;
                    if (mGrid.isSolidCell(i, j)) {
                        mGrid.mU(i, j) = 0.0f;
                        mGrid.mV(i, j) = 0.0f;//不加会在边界发生奇怪的东西
                        continue;
                    }
                    if (i - 1 >= 0 && !mGrid.isSolidCell(i - 1, j)&& mGrid.isValid(i, j, MACGrid2d::X)&&i<=mGrid.dim[0]) {
                        changeX = mP[i][j] - mP[i - 1][j];
                    }

                    if (j - 1 >= 0 && !mGrid.isSolidCell(i, j - 1)&& mGrid.isValid(i, j, MACGrid2d::Y) && j <= mGrid.dim[1]) {
                        changeY = mP[i][j] - mP[i][j - 1];
                    }
                    mGrid.mU(i, j) -= dt / (Eulerian2dPara::airDensity * mGrid.cellSize) * changeX;
                    mGrid.mV(i, j) -= dt / (Eulerian2dPara::airDensity * mGrid.cellSize) * changeY;

                }

        }

        void Solver::BoundaryConditions()
        {
            FOR_EACH_CELL
            {
                if (mGrid.isSolidCell(i, j))
                {
                    
                    mGrid.mU(i, j) = 0.0f;
                    mGrid.mV(i, j) = 0.0f;
                    mGrid.mD(i, j) = 0.0f;
                    mGrid.mT(i, j) = 0.0f;

                }
            }
        }


    }
}
