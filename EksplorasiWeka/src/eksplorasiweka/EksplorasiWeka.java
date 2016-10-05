/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package eksplorasiweka;

import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Attribute;


import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
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
            data.setClassIndex(data.numAttributes() - 1);
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
    public void resample(double b, double z, int seed){
        try {
            System.out.println(data.toString() + "\n");
            Resample resampleFilter = new Resample();
            
            resampleFilter.setInputFormat(data);
            resampleFilter.setNoReplacement(false);
            resampleFilter.setBiasToUniformClass(b); // Uniform distribution of class
            resampleFilter.setSampleSizePercent(z);
            resampleFilter.setRandomSeed(seed);
            
            data = Filter.useFilter(data,resampleFilter);
            
            /*Random R = new Random();
            data.resample(R);*/
            System.out.println("HASIL RESAMPLE\n\n");
            System.out.println(data.toString() + "\n");
        } catch (Exception ex) {
            Logger.getLogger(EksplorasiWeka.class.getName()).log(Level.SEVERE, null, ex);
        }
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
    
    //BuildClassifier: C4.5
    public Classifier C45(){
        Classifier model = null;
        try {
            data.setClassIndex(data.numAttributes()-1);
            J48 tree = new J48();
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
            data.randomize(new java.util.Random(0));
            int trainSize = (int) Math.round((double) data.numInstances() * percent/100f);
            int testSize = data.numInstances() - trainSize;
            
            Instances train = new Instances(data, 0, trainSize);
            Instances test = new Instances(data, trainSize, testSize);
 
            /*for(int i=0; i<trainSize; i++){
                train.add(data.instance(i));
            }
            for(int i=trainSize; i<data.numInstances(); i++){
                test.add(data.instance(i));
            }*/
 
            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, test);
            System.out.println("PERCENTAGE SPLIT\n\n");
            
            System.out.println(eval.toClassDetailsString("=== Detailed Accuracy By Class ===\n"));
            System.out.println(eval.toMatrixString("=== Confusion Matrix ===\n"));
            System.out.println(eval.toSummaryString("\n=== Summary ===\n", false));
            
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
            System.out.println("Berhasil Load Model\n");
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
        boolean stat = true;
        while(stat){
            System.out.println("\n\nProgram Eksplorasi Weka");
            System.out.println("1. Load data set");
            System.out.println("2. Filter : Resample");
            System.out.println("3. Remove Attribute");
            System.out.println("4. Build Classifier");
            System.out.println("5. 10 Fold Cross Validation");
            System.out.println("6. Percentage Split");
            System.out.println("7. Save model");
            System.out.println("8. Load model");
            System.out.println("8. Exit");
            System.out.print("Pilih Menu : "); 
            int option = scan.nextInt();
            if(option == 1) {
                System.out.println("====LOAD DATA====");
                System.out.println("Pilih data yang akan digunakan:");
                System.out.println("1. Weather - Nominal");
                System.out.println("2. Weather - Kontinu");
                System.out.println("3. Iris");
                System.out.print("Nomor data : ");
                int idData = scan.nextInt();
                if(idData == 1) {
                    file = "data/weather.nominal.arff";
                    testfile = "data/weather.nominal.test.arff";
                }
                else if(idData == 2) {
                    file = "data/weather.numeric.arff";
                    testfile = "data/weather.numeric.test.arff";
                }
                else if(idData == 3){
                    file = "data/iris.arff";
                    testfile = "data/iris.test.arff";
                }
                w.loadFile(file); 
            }else if (option == 2){
                System.out.println("====RESAMPLE====");
                System.out.println("-B 0 = distribution in input data -- 1 = uniform distribution.");
                System.out.print("Masukan nilai B : ");
                int bias = scan.nextInt();
                System.out.println("-S Specify the random number seed (default 1)");
                System.out.print("Masukan nilai S : ");
                int seed = scan.nextInt();
                System.out.println("-Z The size of the output dataset, as a percentage of\n" +
                "  the input dataset (default 100)");
                System.out.println("Masukan nilai Z : ");
                int Z = scan.nextInt();
                w.resample(bias, Z, seed);
            }else if (option == 3){
                //remove attribute
                System.out.println("menghapus atribut? (Y/N)");
                String remove = scan.next();
                
                if(remove.equalsIgnoreCase("Y")){
                    int idx[] = new int[1];
                    System.out.print("Index atribut yang akan dihapus: ");
                    idx[0] = scan.nextInt() - 1;
                    w.removeAttribute(idx);
                }
            } else if(option == 4) {
                System.out.println("====Build Classifier====");
                //create model
                System.out.println("Classifier yang akan digunakan:");
                System.out.println("1. Naive Bayes - Weka");
                System.out.println("2. DT - Weka");
                System.out.println("3. C4.5 - Weka");
                System.out.print("Masukan pilihan : ");
                int pil = scan.nextInt();
                
                if(pil == 1){
                    model = w.naiveBayesClassifier();
                }
                else if(pil == 2){
                    model = w.id3Classifier();
                }
                else if(pil == 3){
                    model = w.C45();
                }
                else if(pil == 4){
                    //belom di implementasi
                }else{
                    System.out.println("Maaf pilihan tidak tersedia");
                }
            }else if(option == 5) {
                //10-fold cross validation
                w.crossValidation(model);
            }else if(option == 6) {
                System.out.print("Masukan nilai percentage split : ");
                double p = scan.nextDouble();
                w.percentageSplit(model, p);
            }else if(option == 7) {
                System.out.println("Ingin menyimpan model? (Y/N)");
                String savemodel = scan.next();
                if(savemodel.equalsIgnoreCase("Y")){
                    System.out.print("Nama file: ");
                    String modelname = scan.next();
                    modelname = "model/" + modelname + ".model";
                    w.saveModel(modelname, model);  
                }
            }else if(option == 8) {
                System.out.print("Nama file yang akan di Load: ");
                String loadmodel = scan.next();
                loadmodel = "model/"+loadmodel;
                model = w.loadModel(loadmodel);
            }
            else {
                stat = false;
                System.out.println("====TERIMAKASIH :D====");
            }
        }
        
        w.classify(testfile, model);
    }
    
}
