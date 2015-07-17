.. _Optimisation:

============
Optimisation
============

The FlightDataAnalyser includes the following Optimisation features.

------------
Node Caching
------------

During processing, nodes are both automatically and manually aligned to match the frequency and offset of other nodes. By default, when a Node is being derived, the frequency and offset of the first available dependency defines the frequency and offset of the derived Node; all subsequent nodes will be automatically aligned to match the frequency and offset of the first. Nodes which are aligned share consistent indexes, e.g. the value of a DerivedParameterNode's array at a certain index refers to the same moment in time as a FlightKeyPointValue with the same index. This is achieved for DerivedParameterNodes and MultistateDerivedParameterNodes by linearly interpolating arrays which is a computationally intensive process.

It is highly probable that the FlightDataAnalyser will attempt to align nodes to the same frequency and offset multiple times as dependencies are often shared between multiple nodes. In these cases, we can avoid repeating the costly alignment process for DerivedParameterNodes and MultistateDerivedParameterNodes by caching the results of alignment. This feature can be toggled by changing the NODE_CACHE setting and is enabled by default as the memory usage difference is roughly 10%, yet the overall execution time reduces by over 20% on average.

Further speed benefits can be gained by changing the NODE_CACHE_OFFSET_DP setting, which is None, i.e. disabled, by default. This setting specifies the offset accuracy of the cache key in decimal places. While the results of cached alignment will no longer be completely accurate, offset interpolation differences are assumed to be of little consequence when increased efficiency is required. For example, if the setting's value is 2, the offset of cache keys will be rounded to two decimal places to increase the likelihood of a cache match. A node named Airspeed with a frequency of 1 and an offset of 0.231 will create a cache key of ('Airspeed', 1, 0.23) and any cache lookup for Airspeed at 1Hz will match if the offset is between 0.15 and 0.25.