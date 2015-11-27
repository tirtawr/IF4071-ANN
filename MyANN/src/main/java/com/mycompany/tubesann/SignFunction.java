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
public class SignFunction implements ActivasionFunction{

    public double calculate(double input) {
        if(input>0){
            return 1;
        }
        else{
            return 0;
        }
    }
    
}
