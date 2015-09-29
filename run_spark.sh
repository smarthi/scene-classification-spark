#!/bin/bash

~/spark-1.4.1-bin-hadoop1/bin/spark-submit --master ip-172-31-31-135.us-west-1.compute.internal  --conf "spark.executor.extraJavaOptions=-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -Djava.library.path=/opt/OpenBLAS/lib -Dorg.nd4j.parallel.enabled=false"" --executor-memory 30G --driver-memory 30g --class org.deeplearning4j.SparkMnist target/scene-classification-spark-1.0-SNAPSHOT.jar

~/spark/bin/spark-submit --master spark://ec2-54-193-87-108.us-west-1.compute.amazonaws.com:7077 --conf "spark.executor.extraJavaOptions=-Dorg.nd4j.parallel.enabled=false spark.driver.extraJavaOptions=-Dorg.nd4j.parallel.enabled=false -Djava.library.path=/opt/OpenBLAS/lib"    --executor-cores 10  --executor-memory 30G --driver-memory 30g --class org.deeplearning4j.SparkMnist target/scene-classification-spark-1.0-SNAPSHOT.jar
 screen ~/spark-1.4.1-bin-hadoop1/bin/spark-submit --master local[*] --conf "spark.executor.extraJavaOptions=-Dorg.nd4j.parallel.enabled=false spark.driver.extraJavaOptions=-Dorg.nd4j.parallel.enabled=false -Djava.library.path=/opt/OpenBLAS/lib"   --driver-memory 20g --class org.deeplearning4j.SparkMnist target/scene-classification-spark-1.0-SNAPSHOT.jar
