% input variables
array[int] of int: readings;
int: start;
int: n = length(readings);
set of int: HOUR = 1..n;
set of int: HOUR0 = 0..n;
int: obj = 1;

% declare temp variable
array [1..n+1] of var 0..infinity: temp;

% constraints to calculate current temperature
constraint temp[1] = start;
constraint forall (index in 2..n+1)(temp[index] = floor((temp[index-1] + readings[index-1])/2));

% print current temperature
output ["temp = \(temp);\n"];

% declare action choices
enum CHOICE = {heat, strongly_heat, cool, strongly_cool, do_nothing};
array[1..n+1] of var CHOICE: choice;
array[1..5] of int: choice_effect = [0,1,-2,4,-5];

% declare cost variable
var 0..infinity: cost;
constraint cost = count(choice, heat)*1 + count(choice, strongly_heat)*5 + count(choice, cool)*3 + count(choice, strongly_cool)*9;

% declare adjustments
constraint forall (index in 1..n+1)(
  forall(choiceIndex in 1..length(choice_effect) where temp[index] + choice_effect[choiceIndex] >=25 \/ temp[index] + choice_effect[choiceIndex] <= 30)(
    temp[index] = temp[index] + choice_effect[choiceIndex] /\
    choice[index] = choice[choiceIndex]
  )
)