/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eksplorasiweka;

import weka.classifiers.Classifier;
import weka.classifiers.Sourcable;
import weka.core.Attribute;
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
    private MyID3[] successors;
    private Attribute attribute;
    private double classValue;
    private double[] distribution;
    private Attribute classAttribute;
    
    public double calculateEntropy(Instances data){
        double[] countClass = new double[data.numClasses()];
        for(int i=0; i<data.numInstances(); i++){
            Instance iTemp = (Instance) data.instance(i);
            countClass[(int) iTemp.classValue()]++;
        }
        double entropy = 0;
        double numData = (double) data.numInstances();
        for(int i=0; i<data.numClasses(); i++){
            if(countClass[i] > 0){
                entropy -= (countClass[i] / numData) * (Utils.log2(countClass[i] / numData));
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
    
    public double calculateIG(Instances data, Attribute att){
        double IG = calculateEntropy(data);
        Instances[] splitData = splitData(data, att);
        for(int i = 0; i < att.numValues(); i++) {
            if(splitData[i].numInstances() > 0){
                IG -= ((double) splitData[i].numInstances() / (double) data.numInstances())
                        * calculateEntropy(splitData[i]);
            }
        }
        return IG; 
    }
    
    public void makeTree(Instances data){
        if (data.numInstances() == 0) {
            attribute = null;
            classValue = Instance.missingValue();
            distribution = new double[data.numClasses()];
            return;
        }
        double IGs[] = new double[data.numAttributes()];
        for (int i=0; i<data.numAttributes()-1; i++){
            IGs[i] = calculateIG(data, data.attribute(i));
        }
        attribute = data.attribute(Utils.maxIndex(IGs));
        if(IGs[attribute.index()] == 0) {
            attribute = null;
            distribution = new double[data.numClasses()];
             
            for(int i=0; i<data.numInstances(); i++){
                Instance iTemp = (Instance) data.instance(i);
                distribution[(int) iTemp.classValue()]++;
            }
            Utils.normalize(distribution);
            classValue = Utils.maxIndex(distribution);
            classAttribute = data.classAttribute();
        }
        else{
            Instances[] splitData = splitData(data, attribute);
            successors = new MyID3[attribute.numValues()];
            for(int i=0; i<attribute.numValues(); i++){
                successors[i] = new MyID3();
                successors[i].makeTree(splitData[i]);
            }
        }
    }
    
    @Override
    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
