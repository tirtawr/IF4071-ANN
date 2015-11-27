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
public class InputNode extends Node{
    private double input;

    InputNode() {
        input = 0;
    }

    public void setInput(double input) {
        this.input = input;
        output = input;
    }

    public InputNode(double input) {
        this.input = input;
    }

    @Override
    public double calculate() {
        
        return input;
    }
    
    
}
