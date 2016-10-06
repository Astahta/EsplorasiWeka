/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eksplorasiweka;

import java.util.Enumeration;
import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;


/**
 *
 * @author FiqieUlya
 */
public class MyID3 extends Classifier 
    implements TechnicalInformationHandler, Sourcable{
    static final long serialVersionUID = -2693678647096322561L;

    private MyID3[] child;

    private Attribute split_attribute;

    private double leaf_class;

    private double[] leaf_distribution;

    private Attribute class_attribute;

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        //result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);
        result.enable(Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);

        return result;
    }

    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);

        // missing class
        data = new Instances(data);
        data.deleteWithMissingClass();

        makeTree(data);
    }

    public int maxAttr(Instances data, Attribute atr) {

        Instances[] maxAttr = splitData(data, atr);
        int[] maxval = new int[atr.numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance temp = data.instance(i);
            maxval[((int) temp.classValue())%maxval.length]++;
        }
        return findmax(maxval);
    }

    public int findmax(int[] input) {
        int max = -1;
        for (int counter = 1; counter < input.length; counter++) {
            if (input[counter] > max) {
                max = counter;
            }
        }
        return max;
    }

    private void makeTree(Instances data) throws Exception {
        if (data.numInstances() == 0) {
            split_attribute = null;
            leaf_class = Double.NaN;
            leaf_distribution = new double[data.numClasses()];
            return;
        }

        // maximum information gain.
        double[] infoGains = new double[data.numAttributes()];
        Enumeration attEnum = data.enumerateAttributes();
        for (int i=0; i<data.numAttributes()-1; i++){
            infoGains[i] = calculateIG(data, data.attribute(i));
        }
        split_attribute = data.attribute(Utils.maxIndex(infoGains));

        // Make leaf if information gain is zero.
        // Otherwise create successors.
        if (Utils.eq(infoGains[split_attribute.index()], 0)) {
            split_attribute = null;
            leaf_distribution = new double[data.numClasses()];
            Enumeration instEnum = data.enumerateInstances();
            while (instEnum.hasMoreElements()) {
                Instance inst = (Instance) instEnum.nextElement();
                leaf_distribution[(int) inst.classValue()]++;
            }
            Utils.normalize(leaf_distribution);
            leaf_class = Utils.maxIndex(leaf_distribution);
            //leaf_class = maxAttr(data, split_attribute);
            class_attribute = data.classAttribute();
        } else {
            Instances[] splitData = splitData(data, split_attribute);
            child = new MyID3[split_attribute.numValues()];
            for (int j = 0; j < split_attribute.numValues(); j++) {
                child[j] = new MyID3();
                child[j].makeTree(splitData[j]);
                if (Utils.eq(splitData[j].numInstances(), 0)) {
                    child[j].leaf_class = maxAttr(data, data.attribute(j));
                }
            }
        }
    }

    

    @Override
    public double[] distributionForInstance(Instance instance)
            throws Exception {
        if (instance.hasMissingValue()) {
            throw new Exception("Can't handle missing value(s)");
        }
        if (split_attribute == null) {
            return leaf_distribution;
        } else {
            return child[(int) instance.value(split_attribute)].distributionForInstance(instance);
        }
    }

    private double calculateIG(Instances data, Attribute att)
            throws Exception {

        double IG = calculateEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int i = 0; i < att.numValues(); i++) {
            if (splitData[i].numInstances() > 0) {
                IG = IG - ((double) splitData[i].numInstances() / (double) data.numInstances())* calculateEntropy(splitData[i]);
            }
        }
        return IG;
    }

    private double calculateEntropy(Instances data) {
        double[] kelas = new double[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            Instance temp = data.instance(i);
            kelas[(int) temp.classValue()]++;
        }
        for (int i = 0; i < data.numClasses(); i++) {
            kelas[i] = kelas[i] / data.numInstances();
        }
        double entropy = 0;
        for (int i = 0; i < data.numClasses(); i++) {
            if (kelas[i] > 0) {
                entropy = entropy - (kelas[i] * Utils.log2(kelas[i]));
            }
        }
        return entropy;
    }

    private Instances[] splitData(Instances data, Attribute att) {

        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
          splitData[i] = new Instances(data, data.numInstances());
        }
        for (int i=0; i<data.numInstances(); i++) {
          Instance insTemp = (Instance) data.instance(i);
          splitData[(int) insTemp.value(att)].add(insTemp);
        }
        for (int i = 0; i < splitData.length; i++) {
          splitData[i].compactify();
        }
        return splitData;
    }

    @Override
    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public String toSource(String string) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
