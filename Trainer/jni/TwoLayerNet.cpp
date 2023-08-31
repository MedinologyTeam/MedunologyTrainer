#include "TwoLayerNet.h"
/*
TwoLayerNet::TwoLayerNet(int inputsiz,int hiddensiz,int outputsiz)
{
	float scale=sqrt(1.0/float(inputsiz));
	W1=MatrixXd::Random(inputsiz,outputsiz);
	W1*=scale;
	//W2=MatrixXd::Random(hiddensiz,hiddensiz);
	//W3=MatrixXd::Random(hiddensiz,hiddensiz);
	//W4=MatrixXd::Random(hiddensiz,outputsiz);
	b1=MatrixXd::Zero(1,outputsiz);
	//b2=MatrixXd::Zero(1,hiddensiz);
	//b3=MatrixXd::Zero(1,hiddensiz);
	//b4=MatrixXd::Zero(1,outputsiz);
	
}
MatrixXd TwoLayerNet::getGradW1(){return gradW1;}
MatrixXd TwoLayerNet::getGradW2(){return gradW2;}
MatrixXd TwoLayerNet::getGradW3(){return gradW3;}
MatrixXd TwoLayerNet::getGradW4(){return gradW4;}

MatrixXd TwoLayerNet::getGradb1(){return gradb1;}
MatrixXd TwoLayerNet::getGradb2(){return gradb2;}
MatrixXd TwoLayerNet::getGradb3(){return gradb3;}
MatrixXd TwoLayerNet::getGradb4(){return gradb4;}

MatrixXd TwoLayerNet::Predict(MatrixXd x)
{
	MatrixXd a=x*W1;//+b1;
	//a=Sigmoid(a);
	/*a=a*W2+b2;
	a=Sigmoid(a);
	a=a*W3+b3;
	a=Sigmoid(a);
	a=a*W4+b4;
	
	//a=Softmax(a);
	return a;
}

void TwoLayerNet::CalcGrad(MatrixXd x,MatrixXd t)
{
	/*
	//forward
	MatrixXd a1=x*W1+b1;
	MatrixXd z1=Sigmoid(a1);
	MatrixXd a2=z1*W2+b2;
	MatrixXd y=Softmax(a2);
	//Backward
	MatrixXd dy=(y-t);
	gradW2=(z1.transpose())*dy;
	gradb2=dy;
	MatrixXd da1t=dy*(W2.transpose());
	double da1=da1t.sum();
	MatrixXd dz1=Sigmoid_Grad(a1)*da1;
	gradW1=(x.transpose())*dz1;
	gradb1=dz1;
	
	gradW1=getGrad(W1,x,t);
	//gradW2=getGrad(W2,x,t);
	//gradb1=getGrad(b1,x,t);
	/*gradb2=getGrad(b2,x,t);
	gradW3=getGrad(W3,x,t);
	gradW4=getGrad(W4,x,t);
	gradb3=getGrad(b3,x,t);
	gradb4=getGrad(b4,x,t);
	
}

double mysquare(double x)
{
	return x*x;
}

double loss(MatrixXd &r,MatrixXd &t)
{
	MatrixXd tmp=r-t;
	tmp.unaryExpr(&mysquare);
	return tmp.sum();
}

#define h 0.0001
MatrixXd TwoLayerNet::getGrad(MatrixXd &subject,MatrixXd &x,MatrixXd &t)
{
	MatrixXd save=subject;
	MatrixXd result=MatrixXd::Zero(save.rows(),save.cols());
	int i,j;
	i=j=0;
	int row=subject.rows();
	int col=subject.cols();
	MatrixXd left,right;
	for(;i<row;++i)
	{
		for(;j<col;++j)
		{
			subject(i,j)+=h;
			MatrixXd res1=Predict(x);
			double l1=loss(res1,t);
			subject(i,j)-=(2*h);
			MatrixXd res2=Predict(x);
			double l2=loss(res2,t);
			subject=save;
			double d=l1-l2;
			d/=(2*h);
			result(i,j)=d;
		}
	}
	return result;
}
*/


	/*
	표준정규분포 만들기
	*/
void buildNormalDist(MatrixXd dest,int sx,int sy)
{
	float * nums=new float[sx*sy];
	double sq2pi=sqrt(2*M_PI);
	sq2pi=1.0/sq2pi;
	float x=-10;
	float dx=20.0/(sx*sy);
	for(int i=0;i<sx*sy;++i)
	{
		nums[i]=sq2pi*exp(-x*x/2);
		x+=dx;
	}
	int nDest,nSour;
	float nTemp;
	srand(time(NULL));
	for(int i=0;i<sx*sy*2;i++)
	{
		nDest = rand()%(sx*sy);
		nSour = rand()%(sx*sy);

		nTemp = nums[nDest];
		nums[nDest] = nums[nSour];
		nums[nSour] = nTemp;
	}
	for(int r=0;r<sy;++r)
	{
		for(int c=0;c<sx;++c)
		{
			dest(r,c)=nums[r*sx+c];
		}
	}
	delete[] nums;
}
	MatrixXd SigmoidLayer::forward(MatrixXd x)
	{
		out=Sigmoid(x);
		return out;
	}
	MatrixXd SigmoidLayer::backward(MatrixXd dout)
	{
		//MatrixXd s=(1.0-out.array()).matrix();
		//logger<<"sig s ssub"<<"r: "<<s.rows()<<"c: "<<s.cols()<<endl;
		//logger<<"out r: "<<out.rows()<<"c: "<<out.cols()<<endl;
		//logger<<"dout r: "<<dout.rows()<<"c: "<<dout.cols()<<endl;
		//s.unaryExpr();
		MatrixXd dx=Sigmoid_Grad(out);
	    dx*=dout(0,0);
		return dx;
	}


	MatrixXd AffineLayer::forward(MatrixXd px)
	{
		x=px;
		MatrixXd out=x*W+b;
		return out;
	}
	MatrixXd AffineLayer::backward(MatrixXd dout)
	{
		MatrixXd dx=dout*(W.transpose());
		dW=(x.transpose())*dout;
		db=dout;
		return dx;
	}
	AffineLayer::AffineLayer(MatrixXd pW,MatrixXd pb)
	{
		W=pW;
		b=pb;
	}

	MatrixXd SoftmaxWithLossLayer::forward(MatrixXd x,MatrixXd pt)
	{
		t=pt;
		y=Softmax(x);
		loss=MatrixXd::Constant(1,1,cross_entropy_error(y,t));
		return loss;
	}
	MatrixXd SoftmaxWithLossLayer::backward(MatrixXd dout)
	{
		MatrixXd dx=y.array()-t.array();
		return dx;
	}
	SoftmaxWithLossLayer::SoftmaxWithLossLayer()
	{
		
	}
	MatrixXd SoftmaxWithLossLayer::forward(MatrixXd x){
		
	}
		TwoLayerNet::TwoLayerNet(int inputsize,int hiddensize,int outputsize)
		{
			W1=MatrixXd::Random(inputsize,hiddensize);
			W2=MatrixXd::Random(hiddensize,outputsize);
			//logger<<W1<<endl;
			//logger<<W2<<endl;
			b1=MatrixXd::Zero(1,hiddensize);
			b2=MatrixXd::Zero(1,outputsize);
			//buildNormalDist(W1,inputsize,hiddensize);
			//buildNormalDist(W2,hiddensize,outputsize);
			float scale1=sqrt(1.0/float(inputsize));
			float scale2=sqrt(1.0/float(hiddensize));
			//logger<<scale1<<endl;
			//logger<<scale2<<endl;
			W1*=scale1;
			W2*=scale2;
			//logger<<W1<<endl;
			//logger<<W2<<endl;
			layers[0]=new AffineLayer(W1,b1);
			layers[1]=new SigmoidLayer();
			layers[2]=new AffineLayer(W2,b2);
			lastLayer=new SoftmaxWithLossLayer();
			//layers[3]=lastLayer;
		}
		TwoLayerNet::~TwoLayerNet()
		{
			delete layers[0];
			delete layers[1];
			delete layers[2];
			//delete layers[3];
			delete lastLayer;
		}
		MatrixXd TwoLayerNet::Predict(MatrixXd x)
		{
			logger<<"predict"<<endl<<"x"<<endl<<x;
			for(int i=0;i<3;++i)
			{
				x=layers[i]->forward(x);
				//logger<<"x"<<endl<<x<<endl;
			}
			return x;
		}
		MatrixXd TwoLayerNet::Loss(MatrixXd x,MatrixXd t)
		{
			MatrixXd y=Predict(x);
			return ((SoftmaxWithLossLayer*)lastLayer)->forward(y,t);
		}
		void TwoLayerNet::Gradient(MatrixXd x,MatrixXd t)
		{
			Loss(x,t);
			MatrixXd dout=MatrixXd::Constant(1,1,1);
			dout=lastLayer->backward(dout);
			//logger<<"last back succ"<<endl;
			//logger<<"r: "<<dout.rows()<<"c: "<<dout.cols()<<endl;
			for(int i=0;i<3;++i)
			{
				//logger<<(2-i)<<"start"<<endl;
				dout=layers[2-i]->backward(dout);
				//logger<<(2-i)<<"end"<<endl;
				//logger<<"dout"<<endl<<dout<<endl;
			}
			//logger<<"layers succ"<<endl;
			gradW1=(((AffineLayer*)layers[0])->dW);
			gradb1=(((AffineLayer*)layers[0])->db);
			gradW2=(((AffineLayer*)layers[2])->dW);
			gradb2=(((AffineLayer*)layers[2])->db);
			//logger<<"gradW1"<<gradW1<<endl<<endl;
			//logger<<"gradW2"<<gradW2<<endl<<endl;
			//logger<<"gradb1"<<gradb1<<endl<<endl;
			//logger<<"gradb2"<<gradb2<<endl<<endl;
			
			
		}
		void TwoLayerNet::updateLayers()
		{
			((AffineLayer*)layers[0])->W=W1;
			((AffineLayer*)layers[0])->b=b1;
			((AffineLayer*)layers[2])->W=W2;
			((AffineLayer*)layers[2])->b=b2;
		}
	
	MatrixXd TwoLayerNet::getGradW1(){return gradW1;}
	MatrixXd TwoLayerNet::getGradW2(){return gradW2;}
	MatrixXd TwoLayerNet::getGradb1(){return gradb1;}
	MatrixXd TwoLayerNet::getGradb2(){return gradb2;}

