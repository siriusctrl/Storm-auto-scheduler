# Overall TODO
1. The Processing speed should different across each bolt, instead of only different at machine level
   - This thing should be adjustable by each application instead of hardcoded into topology or bolt
2. 

# Assumption List about this Simulator

### Bolt
1. We are now assuming only computational intensive job. As we are not simulating a literature 'shared' machine (event from bolt level will be processed one by one by the underlying machine).
2. Setting a good random seed is non-trivial. As all replicas of a ***bolt*** will share exactly the same behavior. This will not be fixed at the moment.
