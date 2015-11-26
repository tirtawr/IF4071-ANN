/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package mynn;

/**
 *
 * @author Tirta
 * @param <X>
 * @param <Y>
 */

public class Tuple<X, Y> { 
  public final X node; 
  public final Y weight; 
  public Tuple(X x, Y y) { 
    this.node = x; 
    this.weight = y; 
  } 
}

