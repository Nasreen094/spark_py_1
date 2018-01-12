The is a spark job that I did as a project. The project details are as follows:

1. Cleanup (Spark Job): A sample dataset of request logs is given in `DataSample.csv`. 
We consider records that have identical `geoinfo` and `timest` as suspicious. Please clean up the sample dataset by filtering out those suspicious request records.

2. Label (Spark Job): A list of POI (point of interest) locations is presented in `POIList.csv`. 
Please assign each request in the `DataSample.csv` to one of those POI locations that has minimum distance to the request location.

3. Analysis (Spark Job):
a) With respect to each POI location, please calculate the average and standard deviation of distance between the POI location to each of its assigned records.
b) Image to draw a circle which is centered at a POI location and includes all records assigned to the POI location. 
Please find out the radius and density (i.e. record count/circle area) for each POI location.

4. Model:
To visualize the popularity of the POI locations, we need to map the density of POIs into a scale from -10 to 10. 
Please give a mathematical model to implement the mapping. 
(Hint: please take consideration about extreme case/outliers, and aim to be more sensitive around mean average to give maximum visual differentiability.)

5. Bonus: try to come up with some reasonable hypotheses regarding POIs, state all assumptions, testing steps and conclusions. 