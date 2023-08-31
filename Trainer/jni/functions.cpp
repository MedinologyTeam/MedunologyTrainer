//Desc: Functions.cpp

#include "platform.h"
#include "functions.h"

double identity_function(double x)
{
	return x;
}

double step_function(double x)
{
	if(x>0)
		return 1;
	return 0;
}

double sigmoid(double x)
{
	return 1.0/(1.0+exp(-x));
}

double sigmoid_grad(double x)
{
	double s=sigmoid(x);
	return (1.0-s)*s;
}

double relu(double x)
{
	if(x>0)return x;
	return 0.0;
}
//FIXME!!!!!!!!!!!!!!!!!!!!!!!!!!
double relu_grad(double x)
{
	return relu(x);
}

double expfunc(double x)
{
	return exp(x);
}


MatrixXd identityFunction(MatrixXd x)
{
	return x;
}

MatrixXd stepFunction(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&step_function);
	return m;
}

MatrixXd Sigmoid(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&sigmoid);
	return m;
}

MatrixXd Sigmoid_Grad(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&sigmoid_grad);
	return m;
}

MatrixXd Relu(MatrixXd x)
{
	MatrixXd m=x;
	m.unaryExpr(&relu);
	return m;
}
/*
MatrixXd Relu_grad()
{
	MatrixXd m=MatrixXd::Zero();
	m.unaryExpr(&relu);
	return m;
	
}
*/
MatrixXd Softmax(MatrixXd x)
{
	MatrixXd y=x.array()-x.maxCoeff();
	y.unaryExpr(&expfunc);
	double s=y.sum();
	return y/s;
}

double cross_entropy_error(MatrixXd y,MatrixXd t)
{
	int i,j;
	//logger<<"t"<<t<<endl;
	t.maxCoeff(&i,&j);
	//logger<<"cee j= "<<j<<" "<<y(0,j)<<endl;
	return -log(y(0,j)+0.001f);
}
//data set ���
//���id=72row 
//��id=29 
float dataset[SYMPTOM_NUM][DISEASE_NUM];

void LoadDataSet()
{
	CsvParser *csvparser = CsvParser_new("/sdcard/symdis.csv", ",", 0);
    CsvRow *row;
	int i,r=0;
    while ((row = CsvParser_getRow(csvparser)) ) {
        const char **rowFields = CsvParser_getFields(row);
        for (i = 0 ; i < CsvParser_getNumFields(row) ; i++) {
        	int data=strtoul(rowFields[i],NULL,10);
          //  printf("%d",data);//�;� �� char ��
            dataset[r][i]=data;
        }
		//printf("\n");
		r++;
        CsvParser_destroy_row(row);
    }
    CsvParser_destroy(csvparser);
}

#include "TwoLayerNet.h"
TwoLayerNet net(SYMPTOM_NUM,HIDDEN_NUM,DISEASE_NUM);
#define learning_rate 0.01
void Train_sub(int i)//i��° ��� �н��Ų��. 
{
	MatrixXd x(1,SYMPTOM_NUM);
	for(int it=0;it<SYMPTOM_NUM;++it)
	{
		x(0,it)=dataset[it][i];
	}
	MatrixXd t=MatrixXd::Zero(1,DISEASE_NUM);
	for(int j=0;j<DISEASE_NUM;++j)
	{
		t(0,j)=0;
	}
	t(0,i)=1;
	//logger<<"gradient start"<<endl;
	net.Gradient(x,t);
	//logger<<"grad end"<<endl;
	MatrixXd gradW1=net.getGradW1();
	MatrixXd gradW2=net.getGradW2();
	MatrixXd gradb1=net.getGradb1();
	MatrixXd gradb2=net.getGradb2();
	//logger<<"get end"<<endl;
	/*MatrixXd gradW3=net.getGradW3();
	MatrixXd gradW4=net.getGradW4();
	MatrixXd gradb3=net.getGradb3();
	MatrixXd gradb4=net.getGradb4();
	*/
	//logger<<"GW1 r: "<<gradW1.rows()<<" c: "<<gradW1.cols()<<endl;
	//logger<<"GW2 r: "<<gradW2.rows()<<" c: "<<gradW2.cols()<<endl;
	//logger<<"Gb1 r: "<<gradb1.rows()<<" c: "<<gradb1.cols()<<endl;
	//logger<<"Gb2 r: "<<gradb2.rows()<<" c: "<<gradb2.cols()<<endl;
	net.W1-=learning_rate*gradW1;
	net.W2-=learning_rate*gradW2;
	//logger<<"b start"<<endl;
	net.b1-=learning_rate*gradb1;
	net.b2-=learning_rate*gradb2;
	//logger<<"update end"<<endl;
	//net.W3-=learning_rate*gradW3;
	//net.W4-=learning_rate*gradW4;
	//net.b3-=learning_rate*gradb3;
	//net.b4-=learning_rate*gradb4;
}


void Train()
{
	int index[DISEASE_NUM];
	for(int i=0;i<DISEASE_NUM;++i)
	{
		index[i]=i;
	}
	int nDest,nSour,nTemp;
	srand(time(NULL));
	for(int a=0;a<2;++a){
		logger<<"shuffle"<<endl;
		for(int i=0;i<100;i++)
		{
			nDest = rand()%DISEASE_NUM;
			nSour = rand()%DISEASE_NUM;

			nTemp = index[nDest];
			index[nDest] = index[nSour];
			index[nSour] = nTemp;
		}
		for(int i=0;i<DISEASE_NUM;++i)
		{
			logger<<"train"<<i<<endl;
			Train_sub(index[i]);
		}
	}
}

 void WriteWeights()
 {
	 logger<<"writing..."<<endl;
 	int row,col;
 	FILE *fp=fopen("/sdcard/weights.txt","w");
 	float data;
 	row=SYMPTOM_NUM,col=HIDDEN_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			data=net.W1(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	row=HIDDEN_NUM,col=DISEASE_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{	
			data=net.W2(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	/*
	row=HIDDEN_NUM,col=HIDDEN_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			data=net.W3(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	row=HIDDEN_NUM,col=DISEASE_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{	
			data=net.W4(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	*/
	row=1,col=HIDDEN_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			data=net.b1(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	
	row=1,col=DISEASE_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			data=net.b2(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	/*
	row=1,col=HIDDEN_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			data=net.b3(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	row=1,col=DISEASE_NUM;
 	fprintf(fp,"%d %d\n",row,col);
	for(int i= 0; i<row;++i)
	{
		for(int j=0;j<col;++j)
		{
			data=net.b4(i,j);
			fprintf(fp,"%lf\n",data);
		}
	}
	*/
 	fclose(fp);
 }

