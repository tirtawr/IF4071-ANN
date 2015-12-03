/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.mycompany.tubesann;

import java.io.Serializable;

/**
 *
 * @author Riady
 */
public class ActivationFunction implements Serializable {

    private static final long serialVersionUID = 0;
    public static double signFunction(double input){
        if(input>0){
            return 1;
        }
        else{
            return -1;
        }
    }
    public static double sigmoidFunction(double input){
 
        return (1/( 1 + Math.pow(Math.E,(-1*input))));
    }
}
