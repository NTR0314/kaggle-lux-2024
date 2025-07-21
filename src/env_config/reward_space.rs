use crate::rules_engine::env::TerminationMode;
use crate::rules_engine::params::{P, FIXED_PARAMS};
use crate::rules_engine::state::{GameResult, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use strum::{EnumIter, IntoEnumIterator};
use RewardSpace::{FinalWinner, MatchWinner, PointsScored, CenterReward, EnergyReward};

#[pyclass(eq, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum RewardSpace {
    FinalWinner,
    MatchWinner,
    PointsScored,
    CenterReward,
    EnergyReward,
}

impl RewardSpace {
    pub fn termination_mode(&self) -> TerminationMode {
        match self {
            FinalWinner => TerminationMode::ThirdMatchWin,
            MatchWinner | PointsScored | CenterReward | EnergyReward => TerminationMode::FinalStep,
        }
    }

    pub fn get_reward(&self, state: &State, result: GameResult) -> [f32; P] {
        match self {
            FinalWinner => Self::from_winner(result.final_winner),
            MatchWinner => Self::from_winner(result.match_winner),
            PointsScored => result.points_scored.map(|scored| scored as f32),
            CenterReward => Self::center(state),
            EnergyReward => Self::energy(state),
        }
    }
    
    fn energy(state: &State) -> [f32; P] {
        let mut rewards = [0.0; P];
        
        for (player_id, player_units) in state.units.iter().enumerate() {
            if !player_units.is_empty() {
                let total_energy: i32 = player_units.iter().map(|unit| unit.energy).sum();
                let avg_energy_ratio = (total_energy as f32 / player_units.len() as f32) / FIXED_PARAMS.max_unit_energy as f32;
                rewards[player_id] = avg_energy_ratio;
            }
        }
        
        rewards
    }

    fn center(state: &State) -> [f32; P] {
        let mut rewards = [0.0; P];
        
        // Calculate map center
        let center_x = FIXED_PARAMS.map_width as f32 / 2.0;
        let center_y = FIXED_PARAMS.map_height as f32 / 2.0;
        let max_distance = (center_x.powi(2) + center_y.powi(2)).sqrt();
        
        // println!("Map center: ({}, {}), Max distance: {}", center_x, center_y, max_distance);
        
        for (player_id, player_units) in state.units.iter().enumerate() {
            if !player_units.is_empty() {
                let mut player_total_reward = 0.0;
                for (unit_id, unit) in player_units.iter().enumerate() {
                    let unit_pos = unit.pos;
                    
                    // Calculate euclidean distance to center
                    let distance_to_center = ((unit_pos.x as f32 - center_x).powi(2) + 
                                            (unit_pos.y as f32 - center_y).powi(2)).sqrt();
                    
                    // Normalize distance and calculate reward (closer = higher reward)
                    let normalized_distance = distance_to_center / max_distance;
                    let unit_reward = 1.0 - normalized_distance;
                    
                    player_total_reward += unit_reward;
                    
                    // Debug prints
                    // println!("Unit {} of Player {} at position: {:?}, Distance: {:.2}, Reward: {:.3}", 
                            //  unit_id, player_id, unit_pos, distance_to_center, unit_reward);
                }
                
                // Average reward across all units for this player
                rewards[player_id] = player_total_reward / player_units.len() as f32;
                // println!("Player {} total reward: {:.3} (avg of {} units)", 
                        //  player_id, rewards[player_id], player_units.len());
            }
        }
        
        rewards
    }
    fn from_winner(winner: Option<u8>) -> [f32; P] {
        if let Some(p) = winner {
            match p {
                0 => [1.0, -1.0],
                1 => [-1.0, 1.0],
                p => panic!("Unexpected winner {p}"),
            }
        } else {
            [0.0, 0.0]
        }
    }
}

#[pymethods]
impl RewardSpace {
    fn __str__(&self) -> PyResult<String> {
        let (_, name) = self.__pyo3__repr__().split_once(".").unwrap();
        Ok(name.to_string())
    }

    #[staticmethod]
    fn list() -> Vec<Self> {
        Self::iter().collect()
    }

    #[staticmethod]
    fn from_str(s: &str) -> PyResult<Self> {
        for rs in RewardSpace::iter() {
            if rs.__str__()? == s {
                return Ok(rs);
            }
        }
        Err(PyValueError::new_err(format!("Invalid RewardSpace '{s}'")))
    }
}
