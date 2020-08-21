#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <math.h>
#include "mex.h"
#include "mat.h"
#include "matrix.h"
//#include <windows.h>

/* Test class */
class TimestampAccumulator {
  public:
    TimestampAccumulator(mwSize N, std::string folder);
	TimestampAccumulator(mwSize N);
    void setFilenames(int fidx);
    void loadTimestamps(std::string filename);
	void addTimestampsToArray(double start, double end);
	void freeTimestamps();
	void setTimestampBuffers(size_t length, mxArray *data_0, mxArray *data_1, mxArray *data_2);
    void initializeHistogramBuffer();
    inline bool fileExists(std::string& name);
    void processTimestamps(double start, double end);
    mxArray *getBuffer();
    void setBuffer(mxArray *buffer);
	std::vector<std::string> filenames;
  private:
    std::string folder;
    mwSize N;
    mwSize size_dims [3];
    mxArray *buffer;
	uint32_t *data_0_ptr;
	uint32_t *data_1_ptr;
	uint32_t *data_2_ptr;
	mxArray *data_0;
	mxArray *data_1;
	mxArray *data_2;
	size_t length;
};

TimestampAccumulator::TimestampAccumulator(mwSize N, std::string folder) : N(N), folder(folder) {}

TimestampAccumulator::TimestampAccumulator(mwSize N) : N(N) {}


mxArray *TimestampAccumulator::getBuffer()
{
    return this->buffer;
}

void TimestampAccumulator::setBuffer(mxArray *buffer)
{
	this->size_dims[0] = 4096;
	this->size_dims[1] = this->N;
	this->size_dims[2] = this->N;
    this->buffer = buffer;
}

void TimestampAccumulator::setTimestampBuffers(size_t length, mxArray *data_0, mxArray *data_1, mxArray *data_2)
{
	this->length = length;
	this->data_0 = data_0;
	this->data_1 = data_1;
	this->data_2 = data_2;
	this->data_0_ptr = (uint32_t *)mxGetPr(data_0);
	this->data_1_ptr = (uint32_t *)mxGetPr(data_1);
	this->data_2_ptr = (uint32_t *)mxGetPr(data_2);
}

void TimestampAccumulator::initializeHistogramBuffer()
{
    mwSize n_dims = 3;
    this->size_dims[0] = 4096; 
    this->size_dims[1] = this->N;
    this->size_dims[2] = this->N;
    //mwSize size_dims [3] = {this->N, this->N, 4096}; 
    this->buffer = mxCreateNumericArray(n_dims, this->size_dims, mxDOUBLE_CLASS, mxREAL);

//     double *buffer_ptr = (double *) mxGetPr(buffer);
//     mwSize s1, s2, s3;
//     s1 = size_dims[0];
//     s2 = size_dims[1];
//     s3 = size_dims[2];
//     for(int i=0; i<s1; i++)
//     { 
//         for(int j=0; j<s2; j++) 
//         { 
//             for(int k=0; k<s3; k++) 
//             { 
//                 buffer_ptr[i + j * s1 + k * (s1 * s2)] = 0; // access to element
//             }
//         }   
//     }
}

inline bool TimestampAccumulator::fileExists(std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

void TimestampAccumulator::setFilenames(int fidx)
{
    int idx = 0;
    if (fidx != -1)
    {
        idx = fidx;
    }
    while (true)
    {
        std::stringstream ss;     
        ss << "results/" << this->folder << "/" << "data_" << std::setfill('0')  << std::setw(6) << idx << ".mat";
        std::string filepath = ss.str();

        if (this->fileExists(filepath))
        {
            this->filenames.push_back(filepath);
            idx++;
        }
        else
        {
            break;
        }
        if (fidx != -1)
        {
            break;
        }
    }
}

void TimestampAccumulator::processTimestamps(double start, double end)
{       
    this->loadTimestamps(filenames[0]);
    this->addTimestampsToArray(start, end);
    this->freeTimestamps();
}

void TimestampAccumulator::loadTimestamps(std::string filename) {
    // open mat file
    const char *filename_c_str = filename.c_str();
    MATFile *mfile = matOpen(filename_c_str, "r");
            
    // read 'data_0', 'data_1', 'data_2' variables
    this->data_0 = matGetVariable(mfile, "data_0");
	this->data_1 = matGetVariable(mfile, "data_1");
	this->data_2 = matGetVariable(mfile, "data_2");
	this->length = mxGetNumberOfElements(data_1);
    
    this->data_0_ptr = (uint32_t *)mxGetPr(this->data_0);
	this->data_1_ptr = (uint32_t *)mxGetPr(this->data_1);
	this->data_2_ptr = (uint32_t *)mxGetPr(this->data_2);

    matClose(mfile);
    return;
}

void TimestampAccumulator::freeTimestamps()
{
	mxDestroyArray(this->data_0);
	mxDestroyArray(this->data_1);
	mxDestroyArray(this->data_2);
}

void TimestampAccumulator::addTimestampsToArray(double start, double end) {
	int s1 = (int)this->size_dims[0]; // t
	int s2 = (int)this->size_dims[1]; // y
	int s3 = (int)this->size_dims[2]; // x
	double *buffer_ptr = (double *)mxGetPr(this->buffer);
	for (int k = 0; k < this->length; ++k)
	{
        if (start > double(this->data_0_ptr[k]) / double(data_0_ptr[length - 1] + 1) ||
            end   < double(this->data_0_ptr[k]) / double(data_0_ptr[length - 1] + 1))
        {
            continue;
        }
        
		int ind = (int)floor(double(this->N * (this->N + 1)) * double(this->data_0_ptr[k]) / double(data_0_ptr[length - 1] + 1));
		int ind_y = ind % (this->N);
		int ind_x = (int)floor(double(ind) / double(this->N));
		if (ind_x % 2 == 1)
		{
			ind_y = (s2 - 1) - ind_y;
		}
		if (ind_x >= s3 || ind_y >= s2)
		{
            //mexPrintf("overflow %d %d\n", ind_x, ind_y);
			continue;
		}
		//buffer_ptr[ind_y + ind_x * s2 + this->data_1_ptr[k] * s1*s2] += 1;
		buffer_ptr[ind_y*s1 + ind_x * s1 * s2 + this->data_1_ptr[k]] += 1;
	}
	return;
}

// bins timestamp files from a folder into a histogram
mxArray* mexcpp(int N, std::string folder, int idx, double start, double end) {
    
    // create class TimestampAccumulator
    TimestampAccumulator timestamp_accumulator(N, folder);
    
    // call setFilenames
    timestamp_accumulator.setFilenames(idx);
    
    // call initializeHistogramBuffer
    timestamp_accumulator.initializeHistogramBuffer();

    // call processTimestamps
    timestamp_accumulator.processTimestamps(start, end);
// 	if (idx == -1)
// 	{
// 		timestamp_accumulator.processTimestamps();
// 	}
// 	else
// 	{
// 		timestamp_accumulator.loadTimestamps(timestamp_accumulator.filenames[idx]);
// 		timestamp_accumulator.addTimestampsToArray();
// 	}
    
    // return buffer to accumulated histogram
    return timestamp_accumulator.getBuffer();

    
    
}

// bins input array into histograms
mxArray* mexcpp2(int N, mxArray *data_0, mxArray *data_1, mxArray *data_2, mxArray *buffer) {


	size_t length = mxGetNumberOfElements(data_1);
	// create class TimestampAccumulator
	TimestampAccumulator timestamp_accumulator(N);

	// call initializeHistogramBuffer
    if (buffer == 0)
    {
        timestamp_accumulator.initializeHistogramBuffer();
    }
    else
    {
        timestamp_accumulator.setBuffer(buffer);
    }

	// call processTimestamps
	timestamp_accumulator.setTimestampBuffers(length, data_0, data_1, data_2);
	timestamp_accumulator.addTimestampsToArray(0, 1);

	// return buffer to accumulated histogram
	return timestamp_accumulator.getBuffer();
}

/* The gateway function. */ 
void mexFunction(int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) { // out = mex(N, folder)

    /* Check for proper number of arguments 
    if(nrhs < 2) {
        mexErrMsgIdAndTxt("MATLAB:mexcpp:nargin",
                          "MEXCPP requires two input arguments.");
    }*/
//     if(nlhs != 1) {
//         mexErrMsgIdAndTxt("MATLAB:mexcpp:nargout",
//                           "MEXCPP requires one output argument.");
//     }

//     /* Check if the input is of proper type */
//     if(!mxIsDouble(prhs[0]) ||                                    // not double
//        mxIsComplex(prhs[0]) ||                                   // or complex
//        !mxIsScalar(prhs[0])) {                                  // or not scalar
//         mexErrMsgIdAndTxt("MATLAB:mexcpp:typeargin",
//                           "First argument has to be double scalar.");
//     }
//     if(!mxIsDouble(prhs[1]) ||                                    // not double
//        mxIsComplex(prhs[1]) ||                                   // or complex
//        !mxIsScalar(prhs[1])) {                                  // or not scalar
//         mexErrMsgIdAndTxt("MATLAB:mexcpp:typeargin",
//                           "Second argument has to be double scalar.");
//     }

	// assume we are passing in the arrays to bin in histograms
	//(N, data_0, data_1, data_2)
    mxArray* buffer = 0;
    
	if (mxIsClass(prhs[1], "uint32"))
	{
        if (nrhs > 4)
        {  
            buffer = (mxArray*) prhs[4];
        }
		double* vin1 = mxGetPr(prhs[0]);
		int N = (int)*vin1;
		mxArray *data_0 = (mxArray*) prhs[1];
		mxArray *data_1 = (mxArray*) prhs[2];
		mxArray *data_2 = (mxArray*) prhs[3];
		mxArray* histograms = mexcpp2(N, data_0, data_1, data_2, buffer);
        if (nlhs > 0)
        {
            plhs[0] = histograms;
        }
		return;
	}

    /* Acquire pointers to the input data */
    double* vin1 = mxGetPr(prhs[0]);
    int N = (int) *vin1;

	int idx = -1;
	if (nrhs > 2)
	{
        double* vin3 = mxGetPr(prhs[2]);
        idx = (int)*vin3;
	}
    
    double start_time = 0, end_time = 0;
    if (nrhs > 4)
    {
        start_time = *((double *) mxGetPr(prhs[3]));
        end_time   = *((double *) mxGetPr(prhs[4]));
    }

    /* get the length of the input string */
    size_t buflen = (mxGetM(prhs[1]) * mxGetN(prhs[1])) + 1;

    /* copy the string data from prhs[0] into a C string input_ buf.    */
    char *input_buf = mxArrayToString(prhs[1]);
    std::string foldername(input_buf);
   
    mxArray* histograms = mexcpp(N, foldername, idx, start_time, end_time);
    plhs[0] = histograms;
}