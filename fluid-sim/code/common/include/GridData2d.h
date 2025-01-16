#pragma once
#ifndef __GRID_DATA_2D_H__
#define __GRID_DATA_2D_H__

#pragma warning(disable: 4244 4267 4996)
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <glm/glm.hpp>

namespace Glb {

	using namespace boost::numeric;

	class GridData2d
	{
	public:
		GridData2d();
		GridData2d(const GridData2d& orig);
		virtual ~GridData2d();
		virtual GridData2d& operator=(const GridData2d& orig);

		// ��Ĭ��ֵ��ʼ������
		virtual void initialize(double dfltValue = 0.0);

		// ���أ�i��j��λ���ϵĿɸ�������
		// ��������ʹ�� GridData2d(i, j) = num �����и�ֵ
		virtual double& operator()(int i, int j);

		// �������꣬���ز�ֵ��õ���ֵ
		// ���ڳ�����Χ�ĵ㣬����Ĭ��ֵ
		virtual double interpolate(const glm::vec2& pt);	// ��ֵ���������ݸ������������귵����������

		// �������ݣ��� mData��ͨ�����ں�����ublas����в���
		ublas::vector<double>& data();

		// �����������꣬���ظõ����ڵ���������
		virtual void getCell(const glm::vec2& pt, int& i, int& j);

		virtual glm::vec2 worldToSelf(const glm::vec2& pt) const;
		double mDfltValue;				// Ĭ��ֵ�����ڳ�ʼ������
		glm::vec2 mMax;					// ��ά�ռ��е�������꣬��ʾ����ĳߴ�
		ublas::vector<double> mData;	// �洢�������ݵ�һά����
		float cellSize;
		int dim[2];
	};

	class GridData2dX : public GridData2d
	{
	public:
		GridData2dX();
		virtual ~GridData2dX();
		virtual void initialize(double dfltValue = 0.0);
		virtual double& operator()(int i, int j);
		virtual glm::vec2 worldToSelf(const glm::vec2& pt) const;
	};

	class GridData2dY : public GridData2d
	{
	public:
		GridData2dY();
		virtual ~GridData2dY();
		virtual void initialize(double dfltValue = 0.0);
		virtual double& operator()(int i, int j);
		virtual glm::vec2 worldToSelf(const glm::vec2& pt) const;
	};

	class CubicGridData2d : public GridData2d
	{
	public:
		CubicGridData2d();
		CubicGridData2d(const CubicGridData2d& orig);
		virtual ~CubicGridData2d();
		virtual double interpolate(const glm::vec2& pt);

	protected:

		double cubic(double q1, double q2, double q3, double q4, double t);
		double interpX(int i, int j, double fracty, double fractx);
		double interpY(int i, int j, double fracty);
	};
}

#endif