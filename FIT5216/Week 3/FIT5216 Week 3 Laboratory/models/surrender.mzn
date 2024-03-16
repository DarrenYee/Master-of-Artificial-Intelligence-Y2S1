% workshop 2
% select a set of trade negotiation party
enum NEGOTIATOR;
NEGOTIATOR: dummy; 

int: l; % minimum party size
int: u; % maximum party size
int: m; % minimum joint ability

array[NEGOTIATOR] of int: honor;
array[NEGOTIATOR,NEGOTIATOR] of int: joint;

var set of NEGOTIATOR: party;
var int: obj;

constraint card(party) >= l /\ card(party) <= u; 

constraint sum (n1 in party, n2 in party where n1 > n2) (joint[n1,n2]) >= m; 


constraint obj = min (n1 in NEGOTIATOR where n1 in party) (honor[n1]);

solve maximize obj;

