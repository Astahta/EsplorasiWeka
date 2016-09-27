/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eksplorasiweka;

import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;


import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


/**
 *
 * @author FiqieUlya
 */
public class EksplorasiWeka {
    private Instances data;
    
    public EksplorasiWeka(){
        data = null;
    }
    
    //load data (arrf dan csv)
    public void loadFile(String data_address){
        try {
            data = ConverterUtils.DataSource.read(data_address);
            System.out.println("LOAD DATA BERHASIL\n\n");
            System.out.println(data.toString() + "\n");   
        } catch (Exception ex) {
            System.out.println("File gagal di-load");
        }     
    }
    
    //remove atribut
    public void removeAttribute(int[] idx){
        try{
            Remove remove = new Remove();
            remove.setAttributeIndicesArray(idx);
            remove.setInputFormat(data);
            data = Filter.useFilter(data, remove);
            System.out.println(data.toString() + "\n");
        } catch (Exception ex){
            System.out.println("Gagal remove attribute!");
        }       
    }
    
    //Filter: Resample
    public void resample(){
        Random R = new Random();
        data.resample(R);
        System.out.println("HASIL RESAMPLE\n\n");
        System.out.println(data.toString() + "\n");   
    }
    
    //Build Classifier: NaiveBayes
    public Classifier naiveBayesClassifier(){
        Classifier model = null;
        try {
            data.setClassIndex(data.numAttributes()-1);
            NaiveBayes p = new NaiveBayes();
            p.buildClassifier(data);
            model = p;
            System.out.println(model.toString());
        } catch (Exception ex) {
            System.out.println("Model NaiveBayes tidak dapat dibuat");
        }
        return model;
    }
    
    //BuildClassifier: DT
    public Classifier id3Classifier(){
        Classifier model = null;
        try {
            data.setClassIndex(data.numAttributes()-1);
            Id3 tree = new Id3();
            tree.buildClassifier(data);
            model = tree;
            System.out.println(model.toString());
        } catch (Exception ex) {
            System.out.println("Tidak bisa berhasil membuat model id3");
        }
        return model;
    }
    
    //10-fold cross validation
    public void crossValidation(Classifier model){
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));
            System.out.println("10 FOLD CROSS VALIDATION\n\n");
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        } catch (Exception ex) {
            System.out.println("10-Fold Cross Validation gagal");
        }
    }
    
    //percentage split
    public void percentageSplit(Classifier model, double percent){
        try {
            int trainSize = (int) Math.round(data.numInstances() * percent/100);
            int testSize = data.numInstances() - trainSize;
            Instances train = new Instances(data, trainSize);
            Instances test = new Instances(data, testSize);;
 
            for(int i=0; i<trainSize; i++){
                train.add(data.instance(i));
            }
            for(int i=trainSize; i<data.numInstances(); i++){
                test.add(data.instance(i));
            }
 
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, test);
            System.out.println("PERCENTAGE SPLIT\n\n");
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
        } catch (Exception ex) {
            System.out.println("Gagal");
        }
    }
    
    //Save Model
    public void saveModel(String modelname, Classifier model){
        try {
            SerializationHelper.write(modelname, model);
            System.out.println("berhasil disave\n");
        } catch (Exception ex) {
            System.out.println("gagal di save\n");
        }
    }
 
    //Load Model
    public Classifier loadModel(String modeladdress){
        Classifier model = null;
        try {
            model  = (Classifier) SerializationHelper.read(modeladdress);
            System.out.println(model.toString());
            System.out.println("berhasil diload\n");
        } catch (Exception ex) {
            System.out.println("tidak bisa diload\n");
        }
        return model;
    }
    
    public void classify(String data_address, Classifier model){
        try {
            Instances test = ConverterUtils.DataSource.read(data_address);
            test.setClassIndex(test.numAttributes()-1);
            System.out.println("#Predictions on user test set#");
            System.out.println("# - actual - predicted - distribution");
            for (int i = 0; i < test.numInstances(); i++) {
                double pred = model.classifyInstance(test.instance(i));
                double[] dist = model.distributionForInstance(test.instance(i));
                System.out.print((i+1) + " - ");
                System.out.print(test.instance(i).toString(test.classIndex()) + " - ");
                System.out.print(test.classAttribute().value((int) pred) + " - ");
                System.out.println(Utils.arrayToString(dist)+ "\n");
            }
        } catch (Exception ex) {
            System.out.println("GAGAL PREDIKSI\n");
        }
    }
   
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        String file = "";
        String testfile = "";
        EksplorasiWeka w = new EksplorasiWeka();
        Scanner scan = new Scanner(System.in);
        Classifier model = null;
         
        System.out.println("Data yang akan digunakan:");
        System.out.println("1. Weather - Nominal");
        System.out.println("2. Weather - Kontinu");
        System.out.println("3. Iris");
        int pil = scan.nextInt();
        if(pil == 1){
            file = "data/weather.nominal.arff";
            testfile = "data/weather.nominal.test.arff";
        }
        else if(pil == 2){
            file = "data/weather.numeric.arff";
            testfile = "data/weather.numeric.test.arff";
        }
        else{
            file = "data/iris.arff";
            testfile = "data/iris.test.arff";
        }
         
        //loadfile
        w.loadFile(file); 
        //filter resample
        w.resample();
                 
        //remove attribute
        System.out.println("menghapus atribut? (Y/N)");
        String remove = scan.next();
        if(remove.equalsIgnoreCase("Y")){
            int idx[] = new int[1];
            System.out.print("Index atribut yang akan dihapus: ");
            idx[0] = scan.nextInt();
            w.removeAttribute(idx);
        }
        
        
         
        //create model
        System.out.println("Classifier yang akan digunakan:");
        System.out.println("1. Naive Bayes");
        System.out.println("2. DT");
        pil = scan.nextInt();
        if(pil == 1){
            model = w.naiveBayesClassifier();
        }
        else if(pil == 2){
            model = w.id3Classifier();
        }
        else if(pil == 3){
            //belom di implementasi
        }
        else if(pil == 4){
            //belom di implementasi
        }else{
            System.out.println("Maaf pilihan tidak tersedia");
        }
        System.out.println(model.toString());
        //10-fold cross validation
        w.crossValidation(model);
        //percentage split
        w.percentageSplit(model, 66);
        //saveModel
        System.out.println("Ingin menyimpan model? (Y/N)");
        String savemodel = scan.next();
        if(savemodel.equalsIgnoreCase("Y")){
            System.out.print("Nama file: ");
            String modelname = scan.next();
            modelname += ".model";
            w.saveModel(modelname, model);   
            System.out.println("SEMENTARA AUTO LOAD");
            model = w.loadModel(modelname);
            //w.crossValidation(model);
        }
        w.classify(testfile, model);
    }
    
}
