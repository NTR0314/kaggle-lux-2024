use crate::rules_engine::env::TerminationMode;
use crate::rules_engine::params::{P, FIXED_PARAMS};
use crate::rules_engine::state::{GameResult, State};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use strum::{EnumIter, IntoEnumIterator};
use RewardSpace::{FinalWinner, MatchWinner, PointsScored, Center};

#[pyclass(eq, rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub enum RewardSpace {
    FinalWinner,
    MatchWinner,
    PointsScored,
    Center,
}

impl RewardSpace {
    pub fn termination_mode(&self) -> TerminationMode {
        match self {
            FinalWinner => TerminationMode::ThirdMatchWin,
            MatchWinner | PointsScored | Center => TerminationMode::FinalStep,
        }
    }

    pub fn get_reward(&self, state: &State, result: GameResult) -> [f32; P] {
        match self {
            FinalWinner => Self::from_winner(result.final_winner),
            MatchWinner => Self::from_winner(result.match_winner),
            PointsScored => result.points_scored.map(|scored| scored as f32),
            Center => Self::center(state),
        }
    }
    
    fn center(state: &State) -> [f32; P] {
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
