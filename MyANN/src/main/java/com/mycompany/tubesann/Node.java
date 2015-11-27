/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.mycompany.tubesann;

/**
 *
 * @author Riady
 */
public class Node {
    private Node[] next;
    private Node[] prev;
    protected double output;

    public double getOutput() {
        return output;
    }
    
    private double[] prevWeight;
    private ActivasionFunction actFunc;
    
    public Node(){
    
    }
    
    public void setNext(Node[] next) {
        this.next = next;
    }

    public void setPrev(Node[] prev) {
        this.prev = prev;
        for(int i=0;i<prev.length;i++){
            Node[] nextNode = new Node[1];
            nextNode[0] = this;
            prev[i].setNext(nextNode);
        }
    }

    public void setPrevWeight(double[] prevWeight) {
        this.prevWeight = prevWeight;
    }

    public void setActFunc(ActivasionFunction actFunc) {
        this.actFunc = actFunc;
    }
    
    
    public double calculate(){
        double ret = 0;
        
        //jumlahin semua prev beserta weightnya
        for(int i=0;i<prev.length;i++){
            ret+=prev[i].calculate() * prevWeight[i];
        }
        //dikenakan activasion function
        output = actFunc.calculate(ret);
        return output;
    }
    
    public void updateWeight(double desiredOutput){
        calculate();
        for(int i=0;i<prev.length;i++){
            prevWeight[i] = prevWeight[i] + MyANN.LEARNINGRATE*(desiredOutput - output)*prev[i].output;
            
        }
        
    }
}
