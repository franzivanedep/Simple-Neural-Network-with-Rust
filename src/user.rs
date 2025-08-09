// user.rs
pub struct User {
    pub username: String,
    pub age: u32,
}

impl User {
    pub fn greet(&self) {
        println!("Hello, {}! You are {} years old.", self.username, self.age);
    }
}
