
class Spout():

    def __init__(self, id:int,
                    incoming_rate:float,
                ) -> None:
        
        self.name = 'spout'
        self.id = id
        self.incoming_rate = incoming_rate
    
    def to_yellow(self, prt): 
        return f'\033[93m {prt}\033[00m'

    def __repr__(self) -> str:
        return self.to_yellow(f'Spout{self.id}')


if __name__ == '__main__':
    a = Spout(1, 1e4)
    print(a)