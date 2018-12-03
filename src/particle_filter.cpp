/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    //Set number of particles
    num_particles = 100;
    
    //Create distributions from GPS data
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    
    //Sample distributions to create particles
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
        weights.push_back(1.0);
    }
    
    is_initialized = true;
    
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    
    default_random_engine gen;
    
    for (int i = 0; i < num_particles; ++i) {
        
        //Calculate new x, y, and theta
        double x_0 = particles[i].x;
        double y_0 = particles[i].y;
        double theta_0 = particles[i].theta;
        
        double pred_x;
        double pred_y;
        double pred_theta;
        
        if (fabs(yaw_rate) < 0.0001) {
            pred_x = x_0 + velocity * delta_t * cos(theta_0);
            pred_y = y_0 + velocity * delta_t * sin(theta_0);
            pred_theta = theta_0;
        }
        else {
            pred_x = x_0 + (velocity / yaw_rate) * (sin(theta_0 + yaw_rate * delta_t) - sin(theta_0));
            pred_y = y_0 + (velocity / yaw_rate) * (cos(theta_0) - cos(theta_0 + yaw_rate * delta_t));
            pred_theta = theta_0 + yaw_rate * delta_t;
        }
        
        //Add gaussian noise
        normal_distribution<double> dist_x(pred_x, std_pos[0]);
        normal_distribution<double> dist_y(pred_y, std_pos[1]);
        normal_distribution<double> dist_theta(pred_theta, std_pos[2]);
        
        //Update particles
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
    
    double curr_dist, prev_dist;
    
    for (int i = 0; i < observations.size(); ++i) {
        
        prev_dist = std::numeric_limits<double>::max();
        
        for (int j = 0; j < predicted.size(); ++j) {
            
            curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            
            if (curr_dist < prev_dist) {

                observations[i].id = j;
                prev_dist = curr_dist;
                
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html
    
    long double weight_total = 0.0;
    
    for (int i = 0; i < num_particles; ++i) {
        
        Particle p_0 = particles[i];
        vector<LandmarkObs> landmarks_in_range;
        vector<LandmarkObs> translated_obs;
        
        //select landmarks within sensor range
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            
            LandmarkObs landmark;
            landmark.id = map_landmarks.landmark_list[j].id_i;
            landmark.x = map_landmarks.landmark_list[j].x_f;
            landmark.y = map_landmarks.landmark_list[j].y_f;
            
            if (dist(landmark.x, landmark.y, p_0.x, p_0.y) <= sensor_range) {
                landmarks_in_range.push_back(landmark);
            }
        }
        
        //convert observations to map coordinates
        
        for (int j = 0; j < observations.size(); ++j) {
            
            LandmarkObs obs = observations[j];
            LandmarkObs trans_obs;
            
            //trans_obs.id = obs.id;
            trans_obs.x = p_0.x + obs.x * cos(p_0.theta) - obs.y * sin(p_0.theta);
            trans_obs.y = p_0.y + obs.x * sin(p_0.theta) + obs.y * cos(p_0.theta);
            trans_obs.id = obs.id;
            
            translated_obs.push_back(trans_obs);
        }
        
        //calculate nearest neighbors between landmarks and observations
        dataAssociation(landmarks_in_range, translated_obs);
        
        //make assignments of landmarks to observations
        vector<int> associations;
        vector<double> sense_x_values;
        vector<double> sense_y_values;
        for (int j = 0; j < translated_obs.size(); ++j) {
            int idx = translated_obs[j].id;
            associations.push_back(landmarks_in_range[idx].id);
            sense_x_values.push_back(landmarks_in_range[idx].x);
            sense_y_values.push_back(landmarks_in_range[idx].y);
        }
        SetAssociations(particles[i], associations, sense_x_values, sense_y_values);
        
        //calculate each observation's contribution to new particle weight
        double exp_portion, diff_x, diff_y;
        double weight_exp = 0.0;
        double P_constant= 1.0/(2.0* M_PI * std_landmark[0] *std_landmark[1]);
        for (int j = 0; j < translated_obs.size(); j++) {
            diff_x =  landmarks_in_range[translated_obs[j].id].x - translated_obs[j].x;
            diff_y =  landmarks_in_range[translated_obs[j].id].y - translated_obs[j].y;
            exp_portion = -(pow(diff_x, 2) / (2.0*std_landmark[0] * std_landmark[0]) + pow(diff_y, 2) / (2.0*std_landmark[1] * std_landmark[1]));
            
            //perform sanity check on exponent portion of probability
            if (fabs(exp_portion) <= 100) {
                weight_exp += exp_portion;
            }
            else {
                weight_exp = -100;
            }
        }
        
        //final weight calculation
        weights[i] = pow(P_constant, translated_obs.size()) * exp(weight_exp);
        
        //sum weights for normalization
        weight_total += weights[i];
    }
    
    //normalize weights
    for(int i = 0; i < num_particles; i++){
        weights[i] /= weight_total;
        particles[i].weight = weights[i];
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    default_random_engine gen;
    discrete_distribution<int> resample_dist(weights.begin(), weights.end());
    vector<Particle> resampled_particles;
    
    for (int i = 0; i < num_particles; ++i) {
        
        resampled_particles.push_back(particles[resample_dist(gen)]);
        
    }
    
    particles = resampled_particles;
    
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                         const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
    
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

