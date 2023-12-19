use crate::chessfrommodel::ChessFromModel;
use crate::chessmodel::ChessModel;
use crate::evaluation::eval_pos;
use crate::mlchess::{best_from, best_to};
use chess::{Board, BoardStatus, ChessMove, Color, Game, MoveGen, Square, ALL_SQUARES};
use std::time::Instant;

static INFINITY: i32 = 10000000;

#[derive(Debug)]
pub struct ChessBot {
    model_from: ChessFromModel,
    model_to: ChessModel,
    game: Game,
    depth: usize,
    moves: usize,
    pieces: usize,
    pub nodes: usize,
    pub ml_calls: usize,
    pub start: Instant,
    pub best_move: ChessMove,
    pub best_score: i32,
}

impl ChessBot {
    pub fn new() -> ChessBot {
        ChessBot {
            model_from: ChessFromModel::load().unwrap(),
            model_to: ChessModel::load().unwrap(),
            game: Game::new_with_board(Board::default()),
            depth: 4,
            moves: 6,
            pieces: 6,
            nodes: 0,
            ml_calls: 0,
            start: Instant::now(),
            best_move: ChessMove::new(Square::E2, Square::E3, None),
            best_score: -10000,
        }
    }

    pub fn reset_board(&mut self) {
        self.game = Game::new_with_board(Board::default());
    }

    pub fn make_move(&mut self, mv: ChessMove) {
        self.game.make_move(mv);
    }

    pub fn sqr_to_str(sqr: i32) -> String {
        let v = "abcdefgh";
        let p = (sqr % 8) as usize;
        String::from(&v[p..=p]) + &(sqr / 8 + 1).to_string()
    }

    fn ensure_legal(
        &self,
        from: &[usize],
        to: &[Vec<usize>],
        idx: usize,
        depth: usize,
    ) -> Option<ChessMove> {
        let mut pick: usize = 0;
        let mut ok = 0;

        loop {
            let m = ChessMove::new(ALL_SQUARES[from[idx]], ALL_SQUARES[to[idx][pick]], None);

            if self.game.current_position().legal(m) {
                ok += 1;

                if ok >= depth {
                    return Some(m);
                }
            }

            if pick >= 63 {
                break;
            }

            pick += 1;
        }

        None
    }

    fn flip_side(side: usize) -> usize {
        1 - side
    }

    fn checkmate_modifier(board: &Board) -> i32 {
        if board.status() == BoardStatus::Checkmate {
            1
        } else {
            0
        }
    }

    pub fn find_best_move(&mut self, depth: usize) -> Option<(Option<ChessMove>, i32)> {
        if depth > self.depth {
            return None;
        }

        if depth == 1 {
            self.nodes = 0;
            self.ml_calls = 0;
            self.start = Instant::now();
        }

        let board = &self.game.current_position();
        let mut legals = MoveGen::new_legal(board);

        if legals.len() < 1 {
            let modifier = if board.status() == BoardStatus::Checkmate {
                1
            } else {
                0
            };
            return Some((None, INFINITY / 3 * modifier));
        }

        let side: usize = if self.game.side_to_move() == Color::White {
            0
        } else {
            1
        };
        let out_from = &best_from(&self.model_from, &[board])[0][side];
        let out_to = &best_to(&self.model_to, &[board], &out_from[0..self.pieces]);

        self.ml_calls += 2;

        let mut score = -INFINITY;
        let mut best: (i32, i32) = (0, 0);

        if depth <= self.depth {
            for i in 0..self.pieces {
                for i2 in 0..self.moves {
                    if let Some(mv) = self.ensure_legal(out_from, out_to, i, i2 + 1) {
                        self.nodes += 1;
                        let new_board = board.make_move_new(mv);

                        if depth == self.depth {
                            let this_eval = eval_pos(&new_board, Self::flip_side(side))
                                + INFINITY / 2 * Self::checkmate_modifier(&new_board);
                            if this_eval > score {
                                best = (i as i32, i2 as i32);
                                score = this_eval;
                            }
                        } else if let Some(res) = self.find_best_move(depth + 1) {
                            if res.1 > score {
                                best = (i as i32, i2 as i32);
                                score = res.1;
                            }
                        }
                    }
                }
            }
        }
        if let Some(mv) = self.ensure_legal(out_from, out_to, best.0 as usize, best.1 as usize + 1) {
            if depth == 1 {
                self.best_move = mv;
                self.best_score = -score;
            }
            return Some((Some(mv), -score));
        } else if depth == 1 {
            if let Some(mv) = self.ensure_legal(out_from, out_to, 0, 1) {
                self.best_move = mv;
                self.best_score = -score;
                return Some((Some(mv), -score));
            } else {
                self.best_move = legals.next().unwrap();
                self.best_score = -100000000;
                return Some((Some(self.best_move), -100000000));
            }
        }

        None
    }
}
