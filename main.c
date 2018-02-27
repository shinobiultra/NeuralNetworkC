#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BINARY_STEP 0
#define SIGMOID 1

#define LAYERS 3

typedef struct NN {
    int *p_Sizes;//int s_InputLayer = 10+1;  +1 for the bias neuron (always equals to 1)
    //int s_HiddenLayer1 = 10+1;
    //int s_HiddenLayer2 = 10+1;
    //int s_OutputLayer = 1;
    double *p_Weights[3];
    double **p_Inputs;
    double *InputX1;
    double *Weights1;
    double *InputX2;
    double *Weights2;
    double *InputX3;
    double *Weights3;
    double *Output;

    double *deltas_output;
    double *deltas_hid2;
    double *deltas_hid1;
    double *p_deltas[3];

    double ***weightDeltas;
}NN;

void Delete_Network(NN* ann);

double Binary_Step(double potential){
    if(potential>=0){
        return 1;
    }else{
        return 0;
    }
}

double Soft_Step(double potential){ //Sigmoid
    return (double) 1.0/(double)(1.0 + exp(-potential));
}

double Activation_Function(double potential, int type){ //Possible to add more activation function as needed
    if(type==0){
        return Binary_Step(potential);
    }else if(type==1){
        return Soft_Step(potential);
    }else{
        exit(1);
    }
}

void Initialize_Weights_Random(double *weightsArray, int sizeX, int sizeY){ //Possible to extend to many more dimensions
    static int i,y;
    srand(time(NULL));
    if(sizeY == 1){  // 1 dimensional array  ---- weightsArray[sizeX]
        for(i=0;i<sizeX;i++){
            *(weightsArray+i) = (double)rand()/(double)RAND_MAX*2.0-1.0; //  Array[a][b] = a*sizeof(b) + b; !!!!
        }
    }else{  // 2 dimensional array ---- weightsArray[sizeX][sizeY]
        for(i=0;i<sizeX;i++){
            for(y=0;y<sizeY;y++){
                *(weightsArray+sizeY*i+y) = (double)rand()/(double)RAND_MAX*2.0-1.0; // Array[a][b] = a*sizeof(b) + b; !!!!
            }
        }
    }
}

void Calculate_Layer(double *weightsArray,int sizeX, int sizeY, double *inputs, double *outputs){
    static int i,y;
    static double sum = 0;
    if(sizeY == 1){  // 1 dimensional array  ---- weightsArray[sizeX]
        for(i=0;i<sizeX;i++){
            sum += inputs[i] * *(weightsArray+i);
        }
        *outputs = Activation_Function(sum, SIGMOID);
        sum = 0;
    }else{  // 2 dimensional array ---- weightsArray[sizeX][sizeY]
        for(i=0;i<sizeY;i++){ // -1 we don't wanna overwrite the bias neuron
            for(y=0;y<sizeX;y++){                                   // sizeY vs. sizeX?!?!?!?!?!?!?!?!?!?!?!?!?!?!
                sum += inputs[y] * *(weightsArray+sizeY*y+i);
            }
            //printf("%f\n",sum);
            outputs[i] = Activation_Function(sum,SIGMOID);
            sum = 0;
        }
    }
}

void Initialize_Biases(double *inputs, int sizeX){
    inputs[sizeX-1] = 1;
}

NN* Initialize_Network(int s_InputLayer, int s_HiddenLayer1, int s_HiddenLayer2, int s_OutputLayer){
    NN *ann =(NN*) malloc(sizeof(NN));
    if(!ann){
        fprintf(stderr, "There was an error while initializing the network.");
        exit(3);
    }
    ann->p_Sizes = (int*) malloc(sizeof(int)*4);
    ann->p_Inputs =  malloc(sizeof(double*)*3);

    ann->InputX1 =(double*) malloc(sizeof(double)*s_InputLayer);                      //    double InputX1[10];
    ann->Weights1 =(double*) malloc(sizeof(double)*s_InputLayer*s_HiddenLayer1);      //    double Weights1[10][10];
    ann->InputX2 =(double*) malloc(sizeof(double)*s_HiddenLayer1);                    //    double InputX2[10];
    ann->Weights2 =(double*) malloc(sizeof(double)*s_HiddenLayer1*s_HiddenLayer2);    //    double Weights2[10][10];
    ann->InputX3 =(double*) malloc(sizeof(double)*s_HiddenLayer2);                    //    double InputX3[10];
    ann->Weights3 =(double*) malloc(sizeof(double)*s_HiddenLayer2*s_OutputLayer);     //    double Weights3[10];
    ann->Output =(double*) malloc(sizeof(double)*s_OutputLayer);                      //    double Output;
    ann->deltas_output =(double*) calloc(s_OutputLayer, sizeof(double));
    ann->deltas_hid2 =(double*) calloc(s_HiddenLayer2, sizeof(double));
    ann->deltas_hid1 =(double*) calloc(s_HiddenLayer1, sizeof(double));
    ann->weightDeltas =(double***) malloc(LAYERS*sizeof(double**));
    if(ann->p_Sizes && ann->p_Inputs && ann->InputX1 && ann->InputX2 && ann->InputX3 && ann->Weights1 && ann->Weights2
                                        && ann->Weights3 && ann->Output && ann->deltas_output && ann->deltas_hid2
                                        && ann->deltas_hid1 && ann->weightDeltas){
        ann->p_Sizes[0] = s_InputLayer;
        ann->p_Sizes[1] = s_HiddenLayer1;
        ann->p_Sizes[2] = s_HiddenLayer2;
        ann->p_Sizes[3] = s_OutputLayer;

        ann->p_Weights[0] = ann->Weights1;
        ann->p_Weights[1] = ann->Weights2;
        ann->p_Weights[2] = ann->Weights3;

        ann->p_Inputs[0] = ann->InputX1;
        ann->p_Inputs[1] = ann->InputX2;
        ann->p_Inputs[2] = ann->InputX3;

        ann->p_deltas[0] = ann->deltas_hid1;
        ann->p_deltas[1] = ann->deltas_hid2;
        ann->p_deltas[2] = ann->deltas_output;

        Initialize_Weights_Random(ann->Weights1,ann->p_Sizes[0],ann->p_Sizes[1]);
        Initialize_Weights_Random(ann->Weights2,ann->p_Sizes[1],ann->p_Sizes[2]);
        Initialize_Weights_Random(ann->Weights3,ann->p_Sizes[2],ann->p_Sizes[3]);

        Initialize_Biases(ann->InputX1,ann->p_Sizes[0]);
        Initialize_Biases(ann->InputX2,ann->p_Sizes[1]);
        Initialize_Biases(ann->InputX3,ann->p_Sizes[2]);

        for(int i=0;i<LAYERS;i++){                                                          /*!! HARDCODED !!*/
            ann->weightDeltas[i] = malloc(s_HiddenLayer1*s_HiddenLayer2*sizeof(double*));
            for(int j=0;j<5;j++){
                ann->weightDeltas[i][j] = calloc(s_HiddenLayer1,sizeof(double));
            }
        }

        return ann;
    }else{
        fprintf(stderr, "There was an error while allocating memory for NN's inner layers.");
        Delete_Network(ann);
        exit(3);
    }
}

void Run_Network(NN* ann){
    Calculate_Layer(ann->Weights1,ann->p_Sizes[0],ann->p_Sizes[1],ann->InputX1,ann->InputX2);
    Calculate_Layer(ann->Weights2,ann->p_Sizes[1],ann->p_Sizes[2],ann->InputX2,ann->InputX3);
    Calculate_Layer(ann->Weights3,ann->p_Sizes[2],ann->p_Sizes[3],ann->InputX3,ann->Output);
}

void Delete_Network(NN* ann){
    free(ann->InputX1);
    free(ann->InputX2);
    free(ann->InputX3);
    free(ann->Weights1);
    free(ann->Weights2);
    free(ann->Weights3);
    free(ann->Output);
    free(ann->p_Sizes);
    free(ann->p_Inputs);
    free(ann->deltas_output);
    free(ann->deltas_hid2);
    free(ann->deltas_hid1);

    for(int i=0;i<LAYERS;i++){                                                          /*!! HARDCODED !!*/
        for(int j=0;j<5;j++){
            free(ann->weightDeltas[i][j]);
        }
        free(ann->weightDeltas[i]);
    }
    free(ann->weightDeltas);
    free(ann);
}

double Calculate_Delta_Output(NN* ann, int nNeuron, double expected, double gamma){
    double result = ann->Output[nNeuron];
    double delta = gamma * result * (1 - result) * (expected - result);
    return delta;
}

double Calculate_Weight_Delta(double delta, double result, double learningRate){
    return learningRate * delta * result;
}

double Calculate_Delta_Hidden(NN* ann, int nNeuron, int layer, double deltas[], int s_deltas, double gamma){
    double sum = 0, delta;
    double result = ann->p_Inputs[layer][nNeuron];
    for(int i=0;i<s_deltas;i++){
        sum += deltas[i] * *(ann->p_Weights[layer]+ann->p_Sizes[layer+1]*nNeuron+i); // i or nNeuron ?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    }
    delta = gamma * result * (1 - result) * sum;
    return delta;
}

void Change_Weight(NN* ann, double weightDelta, int layer, int x, int y){
    *(ann->p_Weights[layer-1]+ann->p_Sizes[layer-1]*x+y) += weightDelta;
}

void Backpropagate_Output(NN* ann, int layer, double expected, double gamma, double learningRate){
    int s_Output = ann->p_Sizes[layer];
    int s_LastHid = ann->p_Sizes[layer-1];
    double weightDelta;
    for(int i=0;i<s_Output;i++){
        ann->deltas_output[i] = Calculate_Delta_Output(ann,i,expected,gamma);
        for(int x=0;x<s_LastHid;x++){
            weightDelta = Calculate_Weight_Delta(ann->deltas_output[i], ann->p_Inputs[layer-1][x],learningRate);
            //Change_Weight(ann, weightDelta, layer, x, i);
            ann->weightDeltas[layer-1][x][i] = weightDelta;
        }
    }
}

void Backpropagate_Hidden(NN* ann, int layer, double gamma, double learningRate){
    int s_Hidden = ann->p_Sizes[layer];
    int s_LastHid = ann->p_Sizes[layer-1];
    double weightDelta;
    for(int i=0;i<s_Hidden;i++){
        ann->p_deltas[layer-1][i] = Calculate_Delta_Hidden(ann,i,layer,ann->p_deltas[layer],ann->p_Sizes[layer+1],gamma); // layer-1 ?
        for(int x=0;x<s_LastHid;x++){
            weightDelta = Calculate_Weight_Delta(ann->p_deltas[layer-1][i], ann->p_Inputs[layer-1][x],learningRate);
            //Change_Weight(ann, weightDelta, layer, x, i);
            ann->weightDeltas[layer-1][x][i] = weightDelta;
        }
    }
}

void Learn_Weights(NN* ann){
    for(int i=0;i<LAYERS;i++){
        for(int j=0;j<ann->p_Sizes[i];j++){
            for(int k=0;k<ann->p_Sizes[i+1];k++){
                *(ann->p_Weights[i]+ann->p_Sizes[i+1]*j+k)  += ann->weightDeltas[i][j][k];
            }
        }
    }
}

void Report_Weights(NN* ann, int inp){
    printf("Input %d:\n",inp);
    for(int i=0;i<LAYERS;i++){
        printf("------LAYER %d------\n",i);
        for(int j=0;j<ann->p_Sizes[i];j++){
            for(int k=0;k<ann->p_Sizes[i+1];k++){
                printf("%d -> %d : %f ",j,k,*(ann->p_Weights[i]+ann->p_Sizes[i+1]*j+k));
                printf("delta -> %f\n",ann->weightDeltas[i][j][k]);
            }
        }
    }
}

void Train_Network(NN* ann, double input[][2], double output[][1], int inps, int epochs){
    for(int y=0;y<epochs;y++){
        for(int x=0;x<inps;x++){
            for(int i=0;i<ann->p_Sizes[0]-1;i++){
                ann->InputX1[i] = input[x][i];
            }
            Run_Network(ann);
            Backpropagate_Output(ann,3,output[x][0],1,0.03);
            Backpropagate_Hidden(ann,2,1,0.03);
            Backpropagate_Hidden(ann,1,1,0.03);
            Learn_Weights(ann);
            //Report_Weights(ann,x);
        }
    }
    Run_Network(ann);
}

int main()
{
    /*---------------HYPERPARAMETERS----------------*/
    int s_InputLayer = 2+1; // +1 for the bias neuron (always equals to 1)
    int s_HiddenLayer1 = 2+1;
    int s_HiddenLayer2 = 2+1;
    int s_OutputLayer = 1;
    /*---------------------------------------------*/

    /*double input[4][2] = {{1,1},{1,0},{0,1},{0,0}};
    double output[4][1] = {{0},{1},{1},{0}};*/
    double input[4][2] = {{1,1},{1,0},{0,1},{0,0}};//,{1,0},{0,0},{1,1},{0,1}};
    double output[4][1] = {{1},{0},{0},{1}};//,{1},{0},{0},{1}};

    NN *ann = Initialize_Network(s_InputLayer, s_HiddenLayer1, s_HiddenLayer2, s_OutputLayer);

    Train_Network(ann,input,output,4,100000);

    for(int i=0;i<2;i++){
        for(int j=0;j<2;j++){
            ann->InputX1[0] = i;
            ann->InputX1[1] = j;
            Run_Network(ann);
            printf("%d - %d -> %f\n",i,j,ann->Output[0]);
        }
    }

    Run_Network(ann);

    printf("Output: %f\n",*ann->Output);

    Delete_Network(ann);

    return 0;
}
