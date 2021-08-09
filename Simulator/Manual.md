# Assumption List about this Simulator

### Bolt
1. We are now assuming only computational intensive jobs. As we are not simulating a literature 'shared' machine (event from bolt level will be processed one by one by the underlying machine).
2. Setting a good random seed is non-trivial. As all replicas of a ***bolt*** will share exactly the same behavior. This will not be fixed at the moment.


### Spout
1. Assume data incoming rate is evenly distributed across every spout even if there is a fluctuation.