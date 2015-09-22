package org.deeplearning4j;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;

/**
 * Created by agibsonccc on 9/21/15.
 */
public class EvaluationOnScene {
    public static void main(String[] args) throws Exception {
        DataSet data = new DataSet();
        data.load(new File("test.bin"));
        BufferedInputStream bis = new BufferedInputStream(new FileInputStream("model.bin"));
        DataInputStream dis = new DataInputStream(bis);
        INDArray params = Nd4j.read(dis);
        MultiLayerConfiguration conf = MultiLayerConfiguration.fromYaml("conf.yaml");
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.setParameters(params);
        Evaluation evaluation = new Evaluation();
        evaluation.eval(data.getLabels(),network.output(data.getFeatureMatrix()));
        System.out.println(evaluation.stats());
    }


}
