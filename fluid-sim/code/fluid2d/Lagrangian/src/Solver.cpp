#include "Lagrangian/include/Solver.h"
#include "Global.h"
#include <iostream>
#include <algorithm>

namespace FluidSimulation
{

    namespace Lagrangian2d
    {
        Solver::Solver(ParticleSystem2d &ps) : mPs(ps)
        {
        }


        void Solver::solve()
        {
            // TODO
            // Solves the fluid simulation by performing some steps, which may include:
            // 1. compute density 
            // 2. compute press
            // 3. compute accleration
            // 4. update velocity and position
            // 5. check boundary
            // 6. update block id
            // ...
            // 第1步：计算粒子密度
            computeDensity();

            // 第2步：计算粒子压力
            computePressure();

            // 第3步：计算粒子的加速度
            computeAcceleration();

            // 第4步：更新粒子的速度和位置
            updateVelocityAndPosition();

            // 第5步：检查边界条件
            checkBoundary();

            // 第6步：更新粒子区块信息
            updateBlockInfo();
        }
        // 在Solver类中，添加计算Poly6核函数的成员函数
       // Poly6 核函数
        float Solver::W(float dist)
        {
            // 如果距离大于支持半径，返回0
            if (dist >= mPs.supportRadius)
            {
                return 0.0f;
            }

            const float h = mPs.supportRadius;
            const float coeff = 315.0f / (64.0f *3.1415926 * pow(h, 9));  // 常数项

            float distSquared = dist * dist;
            return coeff * pow(h * h - distSquared, 3);
        }


        void Solver::computeDensity()
        {
            // 遍历所有粒子
            for (auto& particle : mPs.particles)
            {
                particle.density = 0.0f;  // 重置粒子密度
                // 遍历其他粒子，计算每个粒子的密度
                for (const auto& neighbor : mPs.particles)
                {
                    if (&particle != &neighbor)
                    {
                        float dist = glm::length(particle.position - neighbor.position);
                        if (dist < mPs.supportRadius)
                        {
                            // 假设每个粒子的质量为1，W是平滑核函数
                            float weight = W(dist);
                            particle.density += weight;  // 计算密度
                        }
                    }
                }
            }
        }
        void Solver::computePressure()
        {
            // 遍历所有粒子
            for (auto& particle : mPs.particles)
            {
                float density_ratio = particle.density / Lagrangian2dPara::density;

                // 确保压力不会变为负数（物理上通常不允许负压力）
                float pressure = Lagrangian2dPara::stiffness * (pow(density_ratio, Lagrangian2dPara::exponent) - 1.0f);

                // 如果压力小于0, 设置为0
                particle.pressure = max(pressure, 0.0f);
            }
        }
        void Solver::computeAcceleration()
        {
            for (auto& particle : mPs.particles)
            {
                glm::vec2 pressureForce(0.0f);
                glm::vec2 viscosityForce(0.0f);

                // 遍历所有粒子，计算力
                for (auto& neighbor : mPs.particles)
                {
                    if (&particle != &neighbor)
                    {
                        float dist = glm::length(particle.position - neighbor.position);
                        if (dist < mPs.supportRadius)
                        {
          
                            glm::vec2 dir = particle.position - neighbor.position;
                            float weight = W(dist);
                            pressureForce += (particle.pressure + neighbor.pressure) / (2.0f * neighbor.density) * weight * glm::normalize(dir);

                            // 粘度力: viscosity * (v_i - v_j) * W
                            viscosityForce += Lagrangian2dPara::viscosity * (neighbor.velocity - particle.velocity) * weight;
                        }
                    }
                }

                // 应用压力力、粘度力和重力
                glm::vec2 acceleration = (pressureForce + viscosityForce) / particle.density;
                acceleration += glm::vec2(Lagrangian2dPara::gravityX, -Lagrangian2dPara::gravityY);

                // 存储计算得到的加速度
                particle.accleration = acceleration;
            }
        }
        void Solver::updateVelocityAndPosition()
        {
            for (auto& particle : mPs.particles)
            {
                // 使用加速度更新速度
                particle.velocity += particle.accleration * Lagrangian2dPara::dt;

                // 限制最大速度
                if (glm::length(particle.velocity) > Lagrangian2dPara::maxVelocity)
                {
                    particle.velocity = glm::normalize(particle.velocity) * Lagrangian2dPara::maxVelocity;
                }

                // 更新位置
                particle.position += particle.velocity * Lagrangian2dPara::dt;
            }
        }
        void Solver::checkBoundary()
        {
            for (auto& particle : mPs.particles)
            {
                // 检查粒子是否超出边界并进行反弹
                if (particle.position.x < mPs.lowerBound.x + Lagrangian2dPara::eps)
                {
                    particle.position.x = mPs.lowerBound.x + Lagrangian2dPara::eps;
                    particle.velocity.x *= -Lagrangian2dPara::velocityAttenuation;
                }
                if (particle.position.x > mPs.upperBound.x - Lagrangian2dPara::eps)
                {
                    particle.position.x = mPs.upperBound.x - Lagrangian2dPara::eps;
                    particle.velocity.x *= -Lagrangian2dPara::velocityAttenuation;
                }

                if (particle.position.y < mPs.lowerBound.y + Lagrangian2dPara::eps)
                {
                    particle.position.y = mPs.lowerBound.y + Lagrangian2dPara::eps;
                    particle.velocity.y *= -Lagrangian2dPara::velocityAttenuation;
                }
                if (particle.position.y > mPs.upperBound.y - Lagrangian2dPara::eps)
                {
                    particle.position.y = mPs.upperBound.y - Lagrangian2dPara::eps;
                    particle.velocity.y *= -Lagrangian2dPara::velocityAttenuation;
                }
            }
        }
        void Solver::updateBlockInfo()
        {
            mPs.updateBlockInfo();
        }






    }
}
