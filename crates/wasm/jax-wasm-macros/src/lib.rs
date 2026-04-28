use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, GenericArgument, ImplItem, ImplItemFn, ItemImpl, Pat, PathArguments, ReturnType, Type,
    TypePath, parse_macro_input,
};

#[proc_macro_attribute]
pub fn export(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemImpl);
    let self_ty = input.self_ty.as_ref().clone();
    let Some(type_ident) = type_ident(&self_ty) else {
        return quote! {
            compile_error!("jax_wasm::export only supports plain type impls");
            #input
        }
        .into();
    };

    let generated_name = type_ident.to_string().to_lowercase();
    let module_ident = format_ident!("__jax_wasm_export_{}", generated_name);
    let import_macro_ident = format_ident!("__jax_wasm_import_{}", generated_name);
    let methods = exported_methods(&input);
    let dispatch_arms = methods.iter().map(dispatch_arm);
    let client_methods = methods
        .iter()
        .map(|method| client_method(method, &module_ident));

    quote! {
        #input

        #[doc(hidden)]
        pub mod #module_ident {
            pub fn shard_id() -> ::std::string::String {
                <super::#type_ident as ::jax::TypedShard>::static_id().to_string()
            }
        }

        impl ::jax_wasm::contract::WasmExport for #type_ident {
            fn shard_id(&self) -> ::std::string::String {
                #module_ident::shard_id()
            }

            fn dispatch(
                &self,
                method: &str,
                request_json: &str,
            ) -> ::core::result::Result<::std::string::String, ::std::string::String> {
                match method {
                    #(#dispatch_arms,)*
                    _ => Err(::std::format!(
                        "Jax WASM: method [{}] is not exported by [{}]",
                        method,
                        stringify!(#type_ident),
                    )),
                }
            }
        }

        #[macro_export]
        macro_rules! #import_macro_ident {
            () => {
                pub struct #type_ident<'a> {
                    jax: &'a ::jax_wasm::guest::Jax,
                }

                impl<'a> ::jax_wasm::guest::WasmShardClient<'a> for #type_ident<'a> {
                    fn from_jax(jax: &'a ::jax_wasm::guest::Jax) -> Self {
                        Self { jax }
                    }
                }

                impl<'a> #type_ident<'a> {
                    #(#client_methods)*
                }
            };
        }
    }
    .into()
}

#[proc_macro]
pub fn import(input: TokenStream) -> TokenStream {
    let path = parse_macro_input!(input as syn::Path);
    let Some(type_ident) = path.segments.last().map(|segment| &segment.ident) else {
        return quote! {
            compile_error!("jax_wasm::import! expects a path to an exported shard type");
        }
        .into();
    };
    let macro_ident = format_ident!(
        "__jax_wasm_import_{}",
        type_ident.to_string().to_lowercase()
    );
    let mut macro_path = path.clone();
    macro_path.segments.pop();
    macro_path.segments.push(syn::PathSegment {
        ident: macro_ident,
        arguments: PathArguments::None,
    });

    quote! {
        #macro_path!();
    }
    .into()
}

struct ExportedMethod {
    ident: syn::Ident,
    args: Vec<ExportedArg>,
    response_ty: Type,
}

struct ExportedArg {
    ident: syn::Ident,
    ty: Type,
}

fn exported_methods(input: &ItemImpl) -> Vec<ExportedMethod> {
    input
        .items
        .iter()
        .filter_map(|item| match item {
            ImplItem::Fn(method) => Some(exported_method(method)),
            _ => None,
        })
        .collect()
}

fn exported_method(method: &ImplItemFn) -> ExportedMethod {
    let args = method
        .sig
        .inputs
        .iter()
        .filter_map(|input| match input {
            FnArg::Receiver(_) => None,
            FnArg::Typed(arg) => {
                let Pat::Ident(ident) = arg.pat.as_ref() else {
                    return None;
                };
                Some(ExportedArg {
                    ident: ident.ident.clone(),
                    ty: arg.ty.as_ref().clone(),
                })
            }
        })
        .collect();

    ExportedMethod {
        ident: method.sig.ident.clone(),
        args,
        response_ty: response_ty(&method.sig.output),
    }
}

fn dispatch_arm(method: &ExportedMethod) -> proc_macro2::TokenStream {
    let method_ident = &method.ident;
    let method_name = method_ident.to_string();
    let arg_count = method.args.len();
    let arg_bindings = method.args.iter().enumerate().map(|(index, arg)| {
        let ident = &arg.ident;
        let ty = &arg.ty;
        quote! {
            let #ident: #ty = <#ty as ::jax_wasm::contract::FromWireValue>::from_wire_value(
                __args
                    .get(#index)
                    .cloned()
                    .ok_or_else(|| ::std::format!(
                        "Jax WASM: method [{}] expected argument [{}]",
                        #method_name,
                        #index,
                    ))?
            )?;
        }
    });
    let arg_idents = method.args.iter().map(|arg| &arg.ident);

    quote! {
        #method_name => {
            let __args = ::jax_wasm::contract::decode_request(request_json)?;
            if __args.len() != #arg_count {
                return Err(::std::format!(
                    "Jax WASM: method [{}] expected [{}] args, got [{}]",
                    #method_name,
                    #arg_count,
                    __args.len(),
                ));
            }
            #(#arg_bindings)*
            ::jax_wasm::contract::IntoWireResult::into_wire_result(self.#method_ident(#(#arg_idents),*))
        }
    }
}

fn client_method(method: &ExportedMethod, module_ident: &syn::Ident) -> proc_macro2::TokenStream {
    let method_ident = &method.ident;
    let method_name = method_ident.to_string();
    let response_ty = &method.response_ty;
    let args = method.args.iter().map(|arg| {
        let ident = &arg.ident;
        let ty = &arg.ty;
        quote! { #ident: #ty }
    });
    let wire_values = method.args.iter().map(|arg| {
        let ident = &arg.ident;
        quote! { ::jax_wasm::contract::ToWireValue::to_wire_value(&#ident)? }
    });

    quote! {
        pub fn #method_ident(
            &self,
            #(#args),*
        ) -> ::core::result::Result<#response_ty, ::std::string::String> {
            let request_json = ::jax_wasm::contract::encode_request(::std::vec![#(#wire_values),*])?;
            let response_json = self.jax.call_raw(
                &$crate::#module_ident::shard_id(),
                #method_name,
                &request_json,
            )?;
            ::jax_wasm::contract::decode_response::<#response_ty>(&response_json)
        }
    }
}

fn response_ty(output: &ReturnType) -> Type {
    match output {
        ReturnType::Default => syn::parse_quote!(()),
        ReturnType::Type(_, ty) => result_ok_ty(ty.as_ref()).unwrap_or_else(|| ty.as_ref().clone()),
    }
}

fn result_ok_ty(ty: &Type) -> Option<Type> {
    let Type::Path(TypePath { path, .. }) = ty else {
        return None;
    };
    let segment = path.segments.last()?;
    match segment.ident.to_string().as_str() {
        "Result" | "JaxResult" => {
            let PathArguments::AngleBracketed(args) = &segment.arguments else {
                return None;
            };
            let GenericArgument::Type(ok_ty) = args.args.first()? else {
                return None;
            };
            Some(ok_ty.clone())
        }
        _ => None,
    }
}

fn type_ident(ty: &Type) -> Option<syn::Ident> {
    let Type::Path(path) = ty else {
        return None;
    };
    path.path
        .segments
        .last()
        .map(|segment| segment.ident.clone())
}
