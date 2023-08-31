#ifndef __TWOLAYERNET
#define __TWOLAYERNET


#include "platform.h"
#include "functions.h"
/*
class TwoLayerNet
{
	public:
		TwoLayerNet(int inputsiz,int hiddensiz,int outputsiz);
		MatrixXd Predict(MatrixXd x);
		MatrixXd W1,W2,W3,W4,b1,b2,b3,b4;
		void CalcGrad(MatrixXd x,MatrixXd t);
		MatrixXd getGradW1();
		MatrixXd getGradW2();
		MatrixXd getGradb1();
		MatrixXd getGradb2();
		MatrixXd getGradW3();
		MatrixXd getGradW4();
		MatrixXd getGradb3();
		MatrixXd getGradb4();
		
	private:
		MatrixXd gradW1;
		MatrixXd gradW2;
		MatrixXd gradb1;
		MatrixXd gradb2;
		MatrixXd gradW3;
		MatrixXd gradW4;
		MatrixXd gradb3;
		MatrixXd gradb4;
		
		MatrixXd getGrad(MatrixXd &subject,MatrixXd &x,MatrixXd &t);
		
};
*/
void buildNormalDist(MatrixXd dest,int sx,int sy);
class Layer
{
	public:
	virtual MatrixXd forward(MatrixXd x)=0;
	virtual MatrixXd backward(MatrixXd dout)=0;
};
class SigmoidLayer:public Layer
{
	public:
	MatrixXd forward(MatrixXd x);
	MatrixXd backward(MatrixXd dout);
	MatrixXd out;
};

/*
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
*/
class AffineLayer:public Layer
{
	public:
	MatrixXd forward(MatrixXd px);
	MatrixXd backward(MatrixXd dout);
	AffineLayer(MatrixXd pW,MatrixXd pb);
	MatrixXd x,W,b,dW,db;
};
/*

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx
*/
double cross_entropy_error(MatrixXd y,MatrixXd t);

class SoftmaxWithLossLayer:public Layer
{
	public:
	MatrixXd forward(MatrixXd x,MatrixXd pt);
	MatrixXd backward(MatrixXd dout);
	SoftmaxWithLossLayer();
	MatrixXd forward(MatrixXd x);
	MatrixXd y,t,loss;
};
/*

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

		*/
		

class TwoLayerNet
	{
		public:
		MatrixXd W1,W2,b1,b2;
		MatrixXd gradW1,gradW2,gradb1,gradb2;
		Layer *layers[3];
		Layer *lastLayer;
		TwoLayerNet(int inputsize,int hiddensize,int outputsize);
		~TwoLayerNet();
		MatrixXd Predict(MatrixXd x);
		MatrixXd Loss(MatrixXd x,MatrixXd t);
		void Gradient(MatrixXd x,MatrixXd t);

		void updateLayers();
		MatrixXd getGradW1();
		MatrixXd getGradW2();
		MatrixXd getGradb1();
		MatrixXd getGradb2();
	};
#endif
