/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.tubesann;

/**
 *
 * @author Tirta
 */
public class SigmoidFunction implements ActivationFunction{

    public double calculate(double input) {
        return (1/( 1 + Math.pow(Math.E,(-1*input))));
    }
    
}
