//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include "predictor.h"
#include <math.h>
#include <stdbool.h>
//
// TODO:Student Information
//
const char *studentName = "Prasannakumaran Dhanasekaran";
const char *studentID   = "A59016520";
const char *email       = "pdhanasekaran@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = { 
  "Static",
  "Gshare",
  "Tournament",
  "Custom" 
};

int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;       // Branch Prediction Type
int verbose;
int address_ghr, predict_local, predict_global, address_pht, address_chooser, pht_value; 
//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//
int ghr;
int* branch_ht = NULL;
int* branch_ht_touranment = NULL;
int* local_ht = NULL;
int* pattern_ht = NULL;
int* chooser = NULL;

#define N 30
#define pcBits 8
#define S 256
#define theta (1.93*(N/4)+14)
#define upper_b 127
#define lower_b -128

struct perceptron { int weights[N+1]; };
struct perceptron perceptrons_taken[S];
int ghr_perceptron[N];

//------------------------------------//
//        Predictor Functions         //
//------------------------------------//
// Initialize the predictor
//

int 
perceptron_predictor(uint32_t pc)
{
  int mask, address_perceptron, sum;
  mask = pow(2, pcBits) - 1;
  address_perceptron = (pc & mask) ^ (ghr & mask);
  sum = perceptrons_taken[address_perceptron].weights[0]; // bias w0
  for (int i = 1; i <= N; i++) 
    sum += perceptrons_taken[address_perceptron].weights[i] * ghr_perceptron[i-1];
  return sum;
}

void
init_predictor()
{
  branch_ht = (uint32_t*)malloc((int) pow(2, ghistoryBits)* sizeof(uint32_t));
  branch_ht_touranment = (uint32_t*)malloc((int) pow(2, ghistoryBits)* sizeof(uint32_t));
  chooser = (uint32_t*)malloc((int) pow(2, ghistoryBits)* sizeof(uint32_t));
  local_ht = (uint32_t*)malloc((int) pow(2, lhistoryBits)* sizeof(uint32_t));
  pattern_ht = (uint32_t*)malloc((int) pow(2, pcIndexBits)* sizeof(uint32_t));

  switch (bpType) {
    case STATIC: break;
    case GSHARE: 
      ghr = 0;
      for (int i = 0; i < (int)pow(2, ghistoryBits); i++) branch_ht[i] = 0;
      break;
    case TOURNAMENT:
      ghr = 0;
      predict_global = 0;
      predict_local = 0;
      for (int i = 0; i < (int)pow(2, ghistoryBits); i++) branch_ht_touranment[i] = 1;
      for (int i = 0; i < (int)pow(2, ghistoryBits); i++) chooser[i] = 1;
      for (int i = 0; i < (int)pow(2, lhistoryBits); i++) local_ht[i] = 1;
      for (int i = 0; i < (int)pow(2, pcIndexBits); i++) pattern_ht[i] = 0;
      break;
    case CUSTOM:
      for (int i = 0; i < S; i++) {
        for (int j = 0; j <= N; j++) {
          perceptrons_taken[i].weights[j] = 0;
        }
      }
      for (int i = 0; i < N; i++) ghr_perceptron[i] = -1;
      break;
    default:
      break;
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint8_t
make_prediction(uint32_t pc)
{
  int address_bht;
  int mask;
  uint8_t prediction;
  int sum;
  // Make a prediction based on the bpType
  switch (bpType) {
    case STATIC:
      return TAKEN;
    case GSHARE:
      mask = pow(2, ghistoryBits) - 1;
      address_bht = (pc & mask) ^ (ghr & mask);
      prediction = (branch_ht[address_bht] <=1) ? NOTTAKEN: TAKEN;
      break;
    case TOURNAMENT:
      // local predictor
      mask = pow(2, pcIndexBits) - 1;
      address_pht = pc & mask;
      pht_value = (pattern_ht[address_pht]) & (int)(pow(2, lhistoryBits) - 1);
      predict_local = (local_ht[pht_value] <= 1) ? NOTTAKEN : TAKEN;
      // global predictor
      mask = pow(2, ghistoryBits) - 1;
      address_bht = ghr & mask;
      predict_global = (branch_ht_touranment[address_bht] <= 1) ? NOTTAKEN : TAKEN;
      // chooser
      mask = pow(2, ghistoryBits) - 1;
      address_chooser = (ghr & mask);
      prediction = chooser[address_chooser] > 1 ? predict_local : predict_global;
      break;
    case CUSTOM:
      sum = perceptron_predictor(pc);
      prediction = (sum >= 0) ? TAKEN : NOTTAKEN;
      break;
    default:
      prediction = NOTTAKEN;
      break;
  }
  return prediction;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//
void 
gshare_train(uint32_t pc, uint8_t outcome) 
{
  int address_bht, mask, value, updated_value;
  mask = pow(2, ghistoryBits) - 1;
  address_bht = (pc & mask) ^ (ghr & mask);
  value = branch_ht[address_bht];
  updated_value = value;
  if (outcome == 0) // 4 cases for weakly not taken, strongly not taken, weakly taken, strongly taken
    {
      if(value != 0) updated_value = value - 1;
    }
    else {
      if (value != 3) updated_value = value + 1;
    }
    branch_ht[address_bht] = updated_value;
  ghr = (ghr << 1) + outcome;
}

void 
tournament_train(uint8_t pc, uint8_t outcome) 
{
  int address_bht, mask;
  int value_local, value_global, value_chooser;
  int updated_value_local, updated_value_global, updated_value_chooser;

  ghr = (ghr << 1) + outcome;
  pattern_ht[address_pht] = (pattern_ht[address_pht] << 1) + (outcome & 1);
  value_chooser = chooser[address_chooser];

  updated_value_chooser = value_chooser;
  if ((outcome == predict_global) && (outcome != predict_local)) 
    {
      if (value_chooser != 0) updated_value_chooser = value_chooser - 1;
    }
    else if ((outcome != predict_global) && (outcome == predict_local)) 
        {
          if (value_chooser != 3) updated_value_chooser = value_chooser + 1;
        }
        chooser[address_chooser] = updated_value_chooser;
  // update GShare
  value_global = branch_ht_touranment[address_ghr];
  updated_value_global = value_global;
  if (outcome == 0)
    {
      if (value_global != 0) updated_value_global = value_global - 1;
    }  
    else {
        if (value_global != 3) updated_value_global = value_global + 1;
      }
      branch_ht_touranment[address_ghr] = updated_value_global;
  // update local
  value_local = local_ht[pht_value];
  updated_value_local = value_local;
  if (outcome == 0)
    {
      if (value_local != 0) updated_value_local = value_local - 1;
    }  
    else {
        if (value_local != 3) updated_value_local = value_local + 1;
      }
    local_ht[pht_value] = updated_value_local;
}

void 
perceptron_train(uint32_t pc, uint8_t outcome) 
{
  int value, updated_value;
  int address_bht, mask;
  int address_perceptron, predicted_sum, predicted, t, ghr_mask;
  mask = pow(2, pcBits) - 1;
  ghr_mask = pow(2, N) - 1;
  address_perceptron = (pc & mask) ^ (ghr & mask);
  predicted_sum = perceptron_predictor(pc);
  predicted = (predicted_sum >= 0) ? TAKEN : NOTTAKEN;
  t = (outcome == TAKEN) ? 1 : -1;
  if ((outcome != predicted) || (abs(predicted_sum) < ceil(theta))) {
    perceptrons_taken[address_perceptron].weights[0] += t;
    for (int i = 1; i <= N; i++) {
      if (perceptrons_taken[address_perceptron].weights[i] <= upper_b && perceptrons_taken[address_perceptron].weights[i] >= lower_b) 
        perceptrons_taken[address_perceptron].weights[i] += t * ghr_perceptron[i-1];
    }
  }
  ghr = (ghr << 1) + outcome;
  for (int i = N-1; i > 0; i--) 
    ghr_perceptron[i] = ghr_perceptron[i-1];
  ghr_perceptron[0] = t;
}

void
train_predictor(uint32_t pc, uint8_t outcome)
{
  switch (bpType) {
    case STATIC: break;
    case GSHARE:
      gshare_train(pc, outcome);
      break;
    case TOURNAMENT:
      tournament_train(pc, outcome);
      break;
    case CUSTOM: 
      perceptron_train(pc, outcome);
      break;
    default: break;
  }
}
